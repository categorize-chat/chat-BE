module.exports = (server, app, sessionMiddleware) => {
  const SocketIO = require("socket.io");
  const Chat = require("./schemas/chat");

  const io = SocketIO(server, { cors: {
    origin: '*', 
    credentials: true
  }, path: "/socket.io" });
  app.set("io", io);
  const room = io.of("/room");
  const chatSocket = io.of("/chat");

  const wrap = (middleware) => (socket, next) =>
    middleware(socket.request, {}, next);
  chatSocket.use(wrap(sessionMiddleware));

  room.on("connection", (socket) => {
    console.log("room 네임스페이스에 접속");
    socket.on("disconnect", () => {
      console.log("room 네임스페이스 접속 해제");
    });
  });

  chatSocket.on("connection", (socket) => {
    console.log("chat 네임스페이스에 접속");

    socket.on("join", (data) => {
      console.log('입장')
      socket.join(data);
      socket.to(data).emit("join", {
        user: "system",
        chat: `${socket.request.nickname}님이 입장하셨습니다.`,
      });
    });

    socket.on("summary", (data) => {
      console.log("Received summary:", data);
      io.of("/chat").to(data.channelId).emit("summary", data);
    });

    socket.on("message", async (data) => {
      try {
        console.log(data);
        const chatData = await Chat.create({
          room: data.roomId,
          nickname: data.nickname,
          content: data.content,
          createdAt: new Date(),
        });
        chatSocket.to(data.roomId).emit("chat", data);
        console.log(chatData);
      } catch (error) {
        console.error(error);
        socket.emit("error", { message: "메시지 저장 중 오류가 발생했습니다." });
      }
    });

    socket.on("disconnect", async () => {
      console.log("chat 네임스페이스 접속 해제");
      const { referer } = socket.request.headers; // 브라우저 주소가 들어있음
      const roomId = new URL(referer).pathname.split("/").at(-1);

      socket.to(roomId).emit("exit", {
        user: "system",
        chat: `${socket.request.nickname}님이 퇴장하셨습니다.`,
      });
    });
  });
};
