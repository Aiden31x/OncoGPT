

"use client";
import { db } from "@/firebase";
import { collection, orderBy, query } from "firebase/firestore";
import { useSession } from "next-auth/react";
import { useCollection } from "react-firebase-hooks/firestore";
import { useEffect, useRef } from "react";
import Message from "./Message";
import { ArrowDownIcon } from "@heroicons/react/24/outline";

type Props = {
    chatId: string;
};

const Chat = ({ chatId }: Props) => {
    const { data: session } = useSession();
    const [messages] = useCollection(
        session &&
            query(
                collection(
                    db,
                    "users",
                    session?.user?.email!,
                    "chats",
                    chatId,
                    "messages"
                ),
                orderBy("createdAt", "asc")
            )
    );
    
    const chatEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (chatEndRef.current) {
            chatEndRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [messages]);

    console.log("messages:", messages?.docs);

    return (
        <div className="flex-1 overflow-y-auto overflow-x-hidden">
            {messages?.empty && (
                <>
                    <p className="mt-10 text-center text-white font-light">
                        Ask Your Question below
                    </p>
                    <ArrowDownIcon className="h-5 w-5 mx-auto mt-5 text-white animate-bounce" />
                </>
            )}
            {messages?.docs.map((message) => (
                <Message key={message.id} message={message.data()} />
            ))}
            <div ref={chatEndRef}></div>
        </div>
    );
};

export default Chat;