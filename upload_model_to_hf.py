import argparse
import os
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Create a Hugging Face Space and upload model + app files")
    parser.add_argument("--model", required=True, help="Path to the model file (pkl)")
    parser.add_argument("--repo-name", default="adhd-noise-classifier", help="Name for the Space repo (no username)")
    parser.add_argument("--token", default=None, help="Hugging Face token (or set HF_TOKEN env var)")
    parser.add_argument("--space-sdk", default="gradio", help="Space SDK to set (gradio or streamlit)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF token not provided. Set HF_TOKEN env var or pass --token.")

    api = HfApi(token=token)
    who = api.whoami()
    username = who.get("name") or who.get("login") or who.get("user")
    if not username:
        raise SystemExit("Could not determine username from token via whoami().")

    repo_id = f"{username}/{args.repo_name}"
    print(f"Creating space repo: {repo_id} (public) ...")
    api.create_repo(repo_id=repo_id, repo_type="space", exist_ok=True, private=False, space_sdk=args.space_sdk)

    # Upload files
    print("Uploading files...")
    api.upload_file(path_or_fileobj=args.model, path_in_repo=os.path.basename(args.model), repo_id=repo_id, repo_type="space")
    api.upload_file(path_or_fileobj="app.py", path_in_repo="app.py", repo_id=repo_id, repo_type="space")
    api.upload_file(path_or_fileobj="requirements-hf.txt", path_in_repo="requirements.txt", repo_id=repo_id, repo_type="space")
    api.upload_file(path_or_fileobj="README_MODEL.md", path_in_repo="README.md", repo_id=repo_id, repo_type="space")

    print("Upload finished.")
    print(f"Visit: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
