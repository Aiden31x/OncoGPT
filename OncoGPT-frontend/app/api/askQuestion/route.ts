import adminDB from "@/firebase-admin";
import { NextApiRequest, NextApiResponse } from "next";
import admin from "firebase-admin";
import axios from "axios";
import { NextResponse } from "next/server";

type Data = {
    answer:string;
}
export async function POST(request:Request, res:NextApiResponse<Data>) {
    const {prompt,chatId,session} = await request.json();
    const userMessage = {
        text: prompt,
        createdAt: admin.firestore.Timestamp.now(),
        user: {
            _id: session.user.email,
            name: session.user.name,
            avatar: session.user.image,
        }
    };
    const response1 = await fetch('http://127.0.0.1:5001/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: prompt }),
    });
    if (!response1.ok) {
        throw new Error(`HTTP error! Status: ${response1.status}`);
    }
    const data = await response1.json();
    console.log("data:", data);
    const message:Message = {
        text: data.answer || "Sorry, I don't understand",
        createdAt: admin.firestore.Timestamp.now(),
        user:{
            _id: "SpaceGPT",
            name: "SpaceGPT",
            avatar: "https://ui-avatars.com/api/?name=AI"
        }

    }
    await adminDB.collection("users").doc(session?.user?.email!).collection("chats").doc(chatId).collection("messages").add(userMessage);
    await adminDB.collection("users").doc(session?.user?.email!).collection("chats").doc(chatId).collection("messages").add(message)
    return  NextResponse.json({answer:message.text as string})
    
}

