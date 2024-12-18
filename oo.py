import os
import subprocess
from datetime import datetime, timedelta

# 깃허브 저장소 URL
remote_repo_url = "https://github.com/nmi344056/AI5.git"

# 저장소 디렉토리 설정
repo_path = "./AI5"

# 저장소 클론 (처음 실행 시 저장소를 복제)
if not os.path.exists(repo_path):
    os.mkdir(repo_path)
    subprocess.run(["git", "clone", remote_repo_url, repo_path])

# 시작 날짜 및 끝 날짜 설정 (예: 2024년 전체)
start_date = datetime(2024, 7, 10)
end_date = datetime(2024, 12, 16)

# 날짜별로 커밋 수행
current_date = start_date
while current_date <= end_date:
    commit_date = current_date.strftime("%Y-%m-%d")
    file_name = f"file_{commit_date}.txt"
    file_path = os.path.join(repo_path, file_name)

    # 파일 생성
    with open(file_path, "w") as f:
        f.write(f"Commit on {commit_date}")

    # Git 명령어 실행
    subprocess.run(["git", "add", "."], cwd=repo_path)
    subprocess.run(
        ["git", "commit", "--date", f"{commit_date}T12:00:00", "-m", f"Commit on {commit_date}"],
        cwd=repo_path,
    )

    # 다음 날짜로 이동
    current_date += timedelta(days=1)

# 변경 사항 푸시
subprocess.run(["git", "push", "origin", "master"], cwd=repo_path)
# subprocess.run(["git", "push", "origin", "main"], cwd=repo_path)

print("깃허브 잔디 전체 채우기 완료!")
