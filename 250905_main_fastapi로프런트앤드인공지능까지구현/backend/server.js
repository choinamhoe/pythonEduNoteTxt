const express = require("express");
require("dotenv").config();
// console.log(process.env.API_KEY);
const api_key = process.env.API_KEY;
const { GoogleGenerativeAI } = require("@google/generative-ai");
const app = express();
app.use(express.json());
const genAI = new GoogleGenerativeAI(api_key);
const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });
app.post("/chat", async (req, res) => {
  const { history } = req.body;
  const conversation = history
    .map((h) => `${h.role === "user" ? "사용자" : "AI"}: ${h.content}`)
    .join("\n");
  console.log(conversation);
  const result = await model.generateContent(conversation);
  console.log(result);
  const message = result.response.text();
  res.status(200).json({ message: message });
});
app.get("/", async (req, res) => {
  return res.status(200).json({ message: "API 호출에 성공하셨습니다." });
});
app.listen(8000, () => {
  console.log("서버 8000번 포트로 실행 중");
});
