import argparse
import os
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Update only the model file in an existing Hugging Face Space")
    parser.add_argument("--model", required=True, help="Path to the model file (pkl)")
    parser.add_argument("--repo-name", required=True, help="Repo name (without username)")
    parser.add_argument("--token", default=None, help="Hugging Face token (or set HF_TOKEN env var)")
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
    print(f"Uploading model to {repo_id} ...")
    api.upload_file(path_or_fileobj=args.model, path_in_repo=os.path.basename(args.model), repo_id=repo_id, repo_type="space")
    print("Model updated.")


if __name__ == "__main__":
    main()
