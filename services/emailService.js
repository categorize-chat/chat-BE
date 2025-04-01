const nodemailer = require('nodemailer');

const transporter = nodemailer.createTransport({
  service: process.env.EMAIL_SERVICE,
  auth: {
    user: process.env.EMAIL_USER,
    pass: process.env.EMAIL_PASSWORD
  }
});

const sendVerificationEmail = async (email, token) => {
  const verificationUrl = `${process.env.CLIENT_URL}/verify-email?token=${token}`;
  
  const mailOptions = {
    from: process.env.EMAIL_USER,
    to: email,
    subject: '이메일 인증을 완료해주세요',
    html: `
      <h1>이메일 인증</h1>
      <p>아래 링크를 클릭하여 이메일 인증을 완료해주세요:</p>
      <a href="${verificationUrl}">${verificationUrl}</a>
      <p>이 링크는 24시간 동안 유효합니다.</p>
    `
  };

  return transporter.sendMail(mailOptions);
};

// 비밀번호 재설정 이메일 전송 함수
const sendPasswordResetEmail = async (email, token) => {
  const resetUrl = `${process.env.CLIENT_URL}/reset-password?token=${token}`;
  
  const mailOptions = {
    from: process.env.EMAIL_USER,
    to: email,
    subject: '비밀번호 재설정 요청',
    html: `
      <h1>비밀번호 재설정</h1>
      <p>아래 링크를 클릭하여 비밀번호를 재설정해주세요:</p>
      <a href="${resetUrl}">${resetUrl}</a>
      <p>이 링크는 1시간 동안 유효합니다.</p>
      <p>비밀번호 재설정을 요청하지 않으셨다면 이 이메일을 무시하셔도 됩니다.</p>
    `
  };

  return transporter.sendMail(mailOptions);
};

module.exports = {
  sendVerificationEmail,
  sendPasswordResetEmail
}; 