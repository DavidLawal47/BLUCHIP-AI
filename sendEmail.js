import nodemailer from "nodemailer";
import dotenv from "dotenv";
dotenv.config();

let transporter = nodemailer.createTransport({
  host: "smtp.gmail.com",
  service: "gmail",
  port: 587,

  auth: {
    user: process.env.Auth_mail,
    pass: process.env.Auth_pass,
  },
});

// TEST TRANSPORTER
transporter.verify((error, success) => {
  if (error) {
    process.env.NODE_ENV === "development" ? console.log(error) : "";
  } else {
    process.env.NODE_ENV === "development"
      ? console.log("Server is ready to take messages", success)
      : "";
  }
});

const sendEmail = async (mailoptions) => {
  try {
    await transporter.sendMail(mailoptions);
    return;
  } catch (error) {
    console.error("Error sending email:", error);
    throw error;
  }
};

export default sendEmail;
