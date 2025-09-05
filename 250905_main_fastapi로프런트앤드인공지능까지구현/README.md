backend 라는 폴더 생성
cd backend # 백엔드라는 폴더로 이동
npm init -y
server.js 파일 생성

<!--  필요한 패키지 설치 -->

npm install express cors dotenv @google/generative-ai

package.json : "start": "nodemon server.js"로 수정하면
npm run start로 서버 실행 가능
