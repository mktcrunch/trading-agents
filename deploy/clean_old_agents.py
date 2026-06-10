#!/usr/bin/env python3
"""
Clean up old, unused Vertex AI Reasoning Engines (Agent Engines).
Keeps only the active ones specified in deploy/agent_engine_ids.env.
"""
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import vertexai
from vertexai.preview import reasoning_engines
from src import config

def clean_agents(dry_run=True):
    project = os.getenv("GCP_PROJECT", "")
    if not project:
        raise SystemExit("Set GCP_PROJECT to your GCP project ID")
    location = os.getenv("GCP_REGION", "us-central1")
    
    vertexai.init(project=project, location=location)
    
    print("=" * 80)
    print(f"CLEANING OLD REASONING ENGINES (Project: {project}, Region: {location})")
    print("=" * 80)
    
    # 1. Load active IDs
    active_ids = set()
    ids_file = Path("deploy/agent_engine_ids.env")
    if ids_file.exists():
        with open(ids_file, "r") as f:
            for line in f:
                if "=" in line:
                    key, val = line.strip().split("=", 1)
                    if "ID" in key and "_RESOURCE" not in key:
                        active_ids.add(val)
                        
    print(f"Active IDs to KEEP (from deploy/agent_engine_ids.env): {list(active_ids)}")
    
    # 2. List all Reasoning Engines
    print("\nFetching deployed Reasoning Engines from Google Cloud...")
    try:
        engines = reasoning_engines.ReasoningEngine.list()
    except Exception as e:
        print(f"Error listing Reasoning Engines: {e}")
        return
        
    if not engines:
        print("No Reasoning Engines found.")
        return
        
    print(f"Found {len(engines)} total Reasoning Engines in your project.")
    
    to_delete = []
    for engine in engines:
        # Extract resource ID from the full resource name
        # Format: projects/{project}/locations/{location}/reasoningEngines/{id}
        resource_name = engine.resource_name
        engine_id = resource_name.split("/")[-1]
        display_name = getattr(engine, "display_name", "Unknown")
        create_time = getattr(engine, "create_time", "Unknown")
        
        if engine_id in active_ids:
            print(f"  [KEEP] ID: {engine_id} | Name: {display_name} (Active)")
        else:
            print(f"  [DELETE] ID: {engine_id} | Name: {display_name} | Created: {create_time}")
            to_delete.append((engine_id, display_name, engine))
            
    if not to_delete:
        print("\n✓ No old Reasoning Engines to clean up!")
        return
        
    print(f"\nFound {len(to_delete)} old/unused Reasoning Engines.")
    
    if dry_run:
        print("\n[DRY RUN] No resources were deleted.")
        print("To delete these old agents, run:")
        print("  python deploy/clean_old_agents.py --execute")
    else:
        print("\nDeleting old Reasoning Engines...")
        for engine_id, display_name, engine in to_delete:
            print(f"  Deleting {display_name} ({engine_id})...")
            try:
                # Force delete the Reasoning Engine and all its child resources (sessions)
                engine.delete(sync=True)
                print(f"  ✓ Deleted {display_name}")
            except Exception as e:
                print(f"  ✗ Failed to delete {display_name}: {e}")
        print("\nCleanup complete!")

if __name__ == "__main__":
    execute = "--execute" in sys.argv
    clean_agents(dry_run=not execute)
