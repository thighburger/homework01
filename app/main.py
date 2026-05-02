import logging
import traceback

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel

try:
    from app.issue import create_github_issue
    from app.spam import check_spam
except ModuleNotFoundError:
    from issue import create_github_issue
    from spam import check_spam

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | "
    "%(filename)s:%(lineno)d (%(funcName)s) | "
    "%(message)s",
)
logger = logging.getLogger("spamcheck")

# FastAPI 기반 웹 앱 생성
# /docs (Swagger UI)에 표기되는 이름
app = FastAPI(title="SpamCheck Web")

# 정적 HTML 서빙: static 안에 파일들을 URL로 접근가능하게 해라
# {URL}/static/…… 으로 접근 가능하게
app.mount("/static", StaticFiles(directory="static"), name="static")

# 메인 페이지 (/) 처리 : “/”로 접속 시 처리할 작업
@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()

class ClassifyRequest(BaseModel):
    text: str   

# classify 요청이 올 때 할 일
# async: 비동기 처리 (서버가 요청 기다리는 동안 다른 요청도 처리 가능
@app.post("/classify")
async def classify(payload: ClassifyRequest):
    text = payload.text
    logger.info(f"CALL /classify | text='{text}' | len={len(text)}")

    try:
        if text == "crash":
            raise RuntimeError("의도적 장애 추가")

        label, score = check_spam(text)
        logger.info(f"OK   /classify | label={label} score={score}")
    except Exception as e:
        logger.exception(
            f"FAIL /classify | text='{text}' | error={type(e).__name__}: {e}"
        )
        tb = traceback.format_exc()
        title = f"[Prod Error] /classify failed: {type(e).__name__}"
        body = (
            "## Summary\n"
            "- endpoint: /classify\n"
            f"- input(text, short): `{text}`\n"
            f"- input length: {len(text)}\n\n"
            "## Exception\n"
            f"- type: {type(e).__name__}\n"
            f"- message: {str(e)}\n\n"
            "## Traceback (line info)\n"
            f"```text\n{tb}\n```"
        )
        create_github_issue(title, body, logger)

        return {"label": "Internal Server Error", "score": -1}

    return {
        "label": label, "score": score
    }

# 실행은 운영 환경의 책임으로 남기기 위해 만들지 X
# http://127.0.0.1:8000 접속
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
