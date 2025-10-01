#!/usr/bin/env python3
"""
Simple script to export WandB run data
"""
import wandb
import pandas as pd

# Your run information
ENTITY = "shichang855-the-university-of-hong-kong"
PROJECT = "brain_tumor_3d_4x80G"
RUN_ID = "7txbdipl"  # Update this to your current run ID

def export_run():
    print("Connecting to WandB...")
    api = wandb.Api()

    run_path = f"{ENTITY}/{PROJECT}/{RUN_ID}"
    print(f"Fetching run: {run_path}")
    run = api.run(run_path)

    print(f"\nRun Name: {run.name}")
    print(f"State: {run.state}")

    # Get history
    print("\nDownloading history...")
    history = run.history()

    print(f"Total steps: {len(history)}")

    # Export full data
    history.to_csv("wandb_full_data.csv", index=False)
    print("✅ Exported to: wandb_full_data.csv")

    # Show key metrics
    if len(history) > 0:
        print("\n" + "="*60)
        print("Latest Metrics:")
        print("="*60)
        latest = history.iloc[-1]

        metrics = [
            'actor/kl_loss',
            'actor/grad_norm',
            'actor/pg_clipfrac',
            'actor/ppo_kl',
            'reward/iou/mean',
            'reward/total/mean'
        ]

        for metric in metrics:
            if metric in latest:
                val = latest[metric]
                if pd.notna(val):
                    # Handle both numeric and string values
                    if isinstance(val, (int, float)):
                        print(f"  {metric:30s}: {val:.6f}")
                    else:
                        print(f"  {metric:30s}: {val}")

    return history

if __name__ == "__main__":
    try:
        history = export_run()
        print("\n✅ Export completed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
