const SocketIO = require("socket.io");
const { removeRoom } = require("./services");

module.exports = (server, app, sessionMiddleware) => {
  const io = SocketIO(server, { cors: {
    origin: '*', 
    credentials: true
}, path: "/socket.io" });
  app.set("io", io);
  const room = io.of("/room");
  const chat = io.of("/chat");

  const wrap = (middleware) => (socket, next) =>
    middleware(socket.request, {}, next);
  chat.use(wrap(sessionMiddleware));

  room.on("connection", (socket) => {
    console.log("room 네임스페이스에 접속");
    socket.on("disconnect", () => {
      console.log("room 네임스페이스 접속 해제");
    });
  });

  chat.on("connection", (socket) => {
    console.log("chat 네임스페이스에 접속");

    socket.on("join", (data) => {
      socket.join(data);
      socket.to(data).emit("join", {
        user: "system",
        chat: `${socket.request.nickname}님이 입장하셨습니다.`,
      });
    });

    socket.on("disconnect", async () => {
      console.log("chat 네임스페이스 접속 해제");
      const { referer } = socket.request.headers; // 브라우저 주소가 들어있음
      const roomId = new URL(referer).pathname.split("/").at(-1);
      const currentRoom = chat.adapter.rooms.get(roomId);

      socket.to(roomId).emit("exit", {
        user: "system",
        chat: `${socket.request.nickname}님이 퇴장하셨습니다.`,
      });
    });
  });
};
