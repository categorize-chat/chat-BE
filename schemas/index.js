const mongoose = require('mongoose');

const {MONGODB_URI} = process.env;
const MONGO_URL = `${MONGODB_URI}`;

const connect = () => {
  mongoose.connect(MONGO_URL, {
    dbName: 'aichat',
    useNewUrlParser: true,
  }).then(() => {
    console.log("몽고디비 연결 성공");
  }).catch((err) => {
    console.error("몽고디비 연결 에러", err);
  });
};

mongoose.connection.on('error', (error) => {
  console.error('몽고디비 연결 에러', error);
});
mongoose.connection.on('disconnected', () => {
  console.error('몽고디비 연결 끊김. 재연결 시도중...');
  connect();
});

module.exports = connect;
