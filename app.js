const express = require('express');
const morgan = require('morgan');
const cookieParser = require('cookie-parser');
const passport = require('passport');
const dotenv = require('dotenv');
const cors = require('cors');

dotenv.config();
const webSocket = require('./socket');
const indexRouter = require('./routes/');
const authRouter = require('./routes/auth');
const connect = require('./schemas');
const passportConfig = require('./passport');
const tempStorage = require('./utils/tempStorage');

const app = express();
passportConfig();
app.set('port', process.env.PORT || 8005);

connect();

(async () => {
  try {
    await tempStorage.getTemp('init');
  } catch (error) {
    console.error('Redis 초기화 오류:', error);
  }
})();

app.use(cors({
  origin: process.env.CLIENT_URL || 'http://localhost:3000',
  credentials: true
}));

app.use(morgan('dev'));
app.use(express.json());
app.use(express.urlencoded({ extended: false }));
app.use(cookieParser());

app.use(passport.initialize());

app.use('/oauth', authRouter);
app.use('/', indexRouter);

app.use((err, req, res, next) => {
  res.locals.message = err.message;
  res.locals.error = process.env.NODE_ENV !== 'production' ? err : {};
  res.status(err.status || 500);
  
  if (err) {
    next(err)
  }
});

const server = app.listen(app.get('port'), () => {
  console.log(app.get('port'), '번 포트에서 대기중');
});

webSocket(server, app);