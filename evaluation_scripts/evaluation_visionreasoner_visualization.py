import argparse
from transformers import Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from qwen_vl_utils import process_vision_info
import torch
import json
from datasets import load_from_disk, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm
import pdb
import os
import re
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoning_model_path", type=str, default="Ricky06662/Seg-Zero-7B")
    parser.add_argument("--segmentation_model_path", type=str, default="facebook/sam2-hiera-large")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--num_parts", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    return parser.parse_args()

def extract_bbox_points_think(output_text, x_factor, y_factor):
    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    if json_match:
        data = json.loads(json_match.group(1))
        pred_bboxes = [[
            int(item['bbox_2d'][0] * x_factor + 0.5),
            int(item['bbox_2d'][1] * y_factor + 0.5),
            int(item['bbox_2d'][2] * x_factor + 0.5),
            int(item['bbox_2d'][3] * y_factor + 0.5)
        ] for item in data]
        pred_points = [[
            int(item['point_2d'][0] * x_factor + 0.5),
            int(item['point_2d'][1] * y_factor + 0.5)
        ] for item in data]
    
    think_pattern = r'<think>([^<]+)</think>'
    think_text = ""
    think_match = re.search(think_pattern, output_text)
    if think_match:
        think_text = think_match.group(1)
    
    return pred_bboxes, pred_points, think_text

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection, union

def compute_bbox_iou(bbox1, bbox2):
    # 计算两个bbox的交集区域
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # 计算交集面积
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # 计算两个bbox的面积
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # 计算并集面积
    union = area1 + area2 - intersection
    
    # 避免除以0
    if union == 0:
        return 0
    
    return intersection / union

def main():
    args = parse_args()
    
    #We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)

    reasoning_model.eval()

    # default processer
    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")
    
    resize_size = 840
    #dataset = load_from_disk(args.test_data_path)['test']
    dataset = load_dataset(args.test_data_path, split='test')
    total_len = len(dataset)
    part_size = total_len // args.num_parts
    start_idx = args.idx * part_size
    end_idx = start_idx + part_size if args.idx < args.num_parts - 1 else total_len
    
    # pdb.set_trace()
    dataset = dataset.select(range(start_idx, end_idx))
    
    if 'bbox' in dataset[0]:
        has_bbox = True
    else:
        has_bbox = False
    
    QUESTION_TEMPLATE = \
        "Please find \"{Question}\" with bboxs and points." \
        "Compare the difference between object(s) and find the most closely matched object(s)." \
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags." \
        "Output the bbox(es) and point(s) inside the interested object(s) in JSON format." \
        "i.e., <think> thinking process here </think>" \
        "<answer>{Answer}</answer>"
    
    messages = []
    id_list = []
    image_list = []
    for item in dataset:
        image = item["image"].convert("RGB")
        message = [{
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "image": image.resize((resize_size, resize_size), PILImage.BILINEAR)
                },
                {   
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(
                        Question=item["text"].lower().strip(".\"?!"),
                        Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410]}]"
                    )    
                }
            ]
        }]
        messages.append(message)
        id_list.append({
            "image_id": item["image_id"],
            "ann_id": item["ann_id"],
            "image": image,
            "mask": item["mask"],
            "img_height": item["img_height"],
            "img_width": item["img_width"],
            "bbox": item["bbox"] if has_bbox else None
        })
        image_list.append(image)

    all_outputs = []
    for i in tqdm(range(0, len(messages), args.batch_size)):
        batch_messages = messages[i:i + args.batch_size]
        batch_id_list = id_list[i:i + args.batch_size]
        batch_image_list = image_list[i:i + args.batch_size]
        # Preparation for inference
        text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        
        image_inputs, video_inputs = process_vision_info(batch_messages)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference: Generation of the output
        generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        batch_output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        
        # pdb.set_trace()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for id_idx in range(len(batch_output_text)):
                try:
                    bboxes, points, think = extract_bbox_points_think(
                                            batch_output_text[id_idx], 
                                            batch_id_list[id_idx]["img_width"]/resize_size, 
                                            batch_id_list[id_idx]["img_height"]/resize_size
                                        )
                except Exception as e:
                    # add penalty in this situation
                    print("Reasoning error: ", e, "Text: ", batch_output_text[id_idx], "ID: ", batch_id_list[id_idx]["image_id"])
                    think = ""
                    intersection = 0
                    union = np.array(batch_id_list[id_idx]["mask"]).sum()
                    bbox_iou = 0.0
                    all_outputs.append({
                        "image_id": batch_id_list[id_idx]["image_id"],
                        "ann_id": batch_id_list[id_idx]["ann_id"],
                        "think": think,
                        "intersection": int(intersection),
                        "union": int(union),
                        "bbox_iou": bbox_iou
                    })
                    continue
                try:
                    segmentation_model.set_image(batch_id_list[id_idx]["image"])
                    mask_all = np.zeros((batch_id_list[id_idx]["img_height"], batch_id_list[id_idx]["img_width"]), dtype=bool)
                except Exception as e:
                    print("Set image error: ", e, batch_id_list[id_idx]["image_id"], batch_id_list[id_idx]["ann_id"])
                    # skip this because the image or mask is not correct
                    continue
                try:
                    for bbox, point in zip(bboxes, points):
                        masks, scores, _ = segmentation_model.predict(
                            point_coords=[point],
                            point_labels=[1],
                            box=bbox
                        )
                        sorted_ind = np.argsort(scores)[::-1]
                        masks = masks[sorted_ind]
                        mask = masks[0].astype(bool)
                        mask_all = np.logical_or(mask_all, mask)
                    gt_mask = np.array(batch_id_list[id_idx]["mask"])
                except Exception as e:
                    print("Segmentation error: ", e, batch_id_list[id_idx]["image_id"], batch_id_list[id_idx]["ann_id"])
                    # skip this because the image or mask is not correct
                    continue
                try:
                    intersection, union = compute_iou(mask_all, gt_mask)
                except Exception as e:
                    print("Image error: ", e)
                    # skip this because the image or mask is not correct
                    continue 
                
                bbox_iou = 0.0
                if has_bbox:
                    try:     
                        gt_bbox = batch_id_list[id_idx]["bbox"]
                        for pred_bbox in bboxes:
                            if compute_bbox_iou(pred_bbox, gt_bbox) > 0.5:
                                bbox_iou = 1.0
                                break
                    except Exception as e:
                        print("Bbox error: ", e, batch_id_list[id_idx]["image_id"], batch_id_list[id_idx]["ann_id"])
                        # skip this because the image or mask is not correct
                        bbox_iou = 0.0

                
                all_outputs.append({
                    "image_id": batch_id_list[id_idx]["image_id"],
                    "ann_id": batch_id_list[id_idx]["ann_id"],
                    "think": think,
                    "intersection": int(intersection),
                    "union": int(union),
                    "bbox_iou": bbox_iou
                })
                
                image = batch_image_list[id_idx]
                ann_id = batch_id_list[id_idx]["ann_id"]
                
                # if int(intersection) / int(union) > 0.3:
                #     continue
                # 修改为1行4列的子图布局
                # print(ann_id, text)
                plt.figure(figsize=(12, 4))
                
                # 第一个子图：原图
                plt.subplot(1, 3, 1)
                plt.imshow(image)
                plt.title('Original Image')
                
                # 第二个子图：原图+bbox
                plt.subplot(1, 3, 2)
                plt.imshow(image)
                # 绘制所有bbox
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    fill=False, edgecolor='red', linewidth=2)
                    plt.gca().add_patch(rect)
                plt.title('Image with Bounding Boxes')
                
                # 第三个子图：mask叠加
                plt.subplot(1, 3, 3)
                plt.imshow(image, alpha=0.6)
                mask_overlay = np.zeros_like(image)
                mask_overlay[mask_all] = [255, 0, 0]
                plt.imshow(mask_overlay, alpha=0.4)
                plt.title('Image with Predicted Mask')
                
                plt.tight_layout()
                
                os.makedirs("visual_results", exist_ok=True)
                #plt.show()
                plt.savefig(os.path.join("visual_results", f"{ann_id}.png"))
                plt.close()  # 关闭图形，释放内存
                
                with open(os.path.join("visual_results", f"{ann_id}.txt"), "w") as f:
                    f.write(f"{int(intersection)/ int(union)}\n{think}")
    
                
        print(f"Processed batch {i//args.batch_size + 1}/{(len(messages) + args.batch_size - 1)//args.batch_size}")
        
        # clean GPU memory
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()

    
    # Modify the output file name, add idx
    output_file = os.path.join(args.output_path, f"output_{args.idx}.json")
    with open(output_file, "w") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
