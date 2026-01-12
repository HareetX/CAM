import argparse
import os
import shutil


def backup_results(backup_dirname: str, dataset: str, workspace: str = "."):
    data_dir = [
        os.path.join(workspace, "data", dataset, "chunks"),
        os.path.join(workspace, "data", dataset, "conversations"),
        os.path.join(workspace, "data", dataset, "questions"),
    ]

    processed_data_dir = os.path.join(workspace, "processed_data", dataset)

    super_embeddings_dir = os.path.join(workspace, "super_embeddings", dataset)
    super_graphs_dir = os.path.join(workspace, "super_graphs", dataset)

    output_dirs = os.path.join(workspace, "output")

    backup_dir = os.path.join(workspace, "backup", backup_dirname)
    os.makedirs(backup_dir, exist_ok=True)

    # Move all relevant directories to the backup directory (Remaining directory structure)
    backup_data_dir = os.path.join(backup_dir, "data", dataset)
    os.makedirs(backup_data_dir, exist_ok=True)
    for dir_path in data_dir:
        if os.path.exists(dir_path):
            dest_path = os.path.join(backup_data_dir, os.path.basename(dir_path))
            # Copy data directory to backup location, not just move it (Only for data)
            shutil.copytree(dir_path, dest_path)

    backup_processed_data_dir = os.path.join(backup_dir, "processed_data", dataset)
    os.makedirs(os.path.dirname(backup_processed_data_dir), exist_ok=True)
    if os.path.exists(processed_data_dir):
        os.rename(processed_data_dir, backup_processed_data_dir)

    backup_super_embeddings_dir = os.path.join(backup_dir, "super_embeddings", dataset)
    os.makedirs(os.path.dirname(backup_super_embeddings_dir), exist_ok=True)
    if os.path.exists(super_embeddings_dir):
        os.rename(super_embeddings_dir, backup_super_embeddings_dir)
    backup_super_graphs_dir = os.path.join(backup_dir, "super_graphs", dataset)
    os.makedirs(os.path.dirname(backup_super_graphs_dir), exist_ok=True)
    if os.path.exists(super_graphs_dir):
        os.rename(super_graphs_dir, backup_super_graphs_dir)

    backup_output_dir = os.path.join(backup_dir, "output")
    os.makedirs(os.path.dirname(backup_output_dir), exist_ok=True)
    if os.path.exists(output_dirs):
        os.rename(output_dirs, backup_output_dir)

    print(f"Backup completed at {backup_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backup dataset results.")
    parser.add_argument("backup_dirname", type=str, help="Name of the backup directory.")
    parser.add_argument("dataset", type=str, help="Name of the dataset to backup.")
    parser.add_argument("--workspace", type=str, default=".", help="Path to the workspace directory.")

    args = parser.parse_args()
    backup_results(args.backup_dirname, args.dataset, args.workspace)
