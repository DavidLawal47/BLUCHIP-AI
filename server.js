import express from "express";
import dotenv from "dotenv";
import globalError from "./utils/globalerror.js";
import { connectDB } from "./config/db.js";
import clientRouter from "./src/user/User.routes.js";
import otpRouter from "./src/otp/otp.routes.js";
import NotificationRouter from "./src/notifications/notifications.routes.js";
import cors from "cors";
import cookieParser from "cookie-parser";
import bodyParser from "body-parser";
import postsRouter from "./src/posts/post.routes.js";
import { fetchAndSaveNews } from "./utils/newfetch.js";
import openaiRouter from "./src/openai/openai.route.js";
dotenv.config();
let url  = process.env.NEWS_STOCKS
const app = express();
const Port = process.env.PORT || 3000;

// Middleware setup
app.use(cors({ origin: process.env.FRONTEND_URL || "http://localhost:5173", credentials: true }));
app.use(express.json({ limit: "50mb" }));
app.use(bodyParser.urlencoded({ extended: true, limit: "10mb" }));
app.use(bodyParser.json({ limit: "200kb" }));
app.use(cookieParser());

// Routes
app.get("/", (req, res) => {
  res.send("Home");
});

app.use("/api/v1/auth", clientRouter);
app.use("/api/v1/otp", otpRouter);
app.use("/api/v1/posts", postsRouter);
app.use("/api/v1/notification", NotificationRouter);
app.use("/api/v1/openai", openaiRouter);


// 404 handler for unknown routes
app.get('/api/news', async (req, res) => {
  try {
    const newsData = await fetchAndSaveNews(url)
    return res.json({data:newsData})
  } catch (error) {
    console.log(error.message);
    
  }
})
app.use((req, res, next) => {
  res.status(404).json({ error: "Not Found" });
});

// Global error handler
app.use(globalError);

// Your API endpoint

// Start server
app.listen(Port, () => {
  connectDB();

  console.log(`Server is running on port ${Port}`);
});
