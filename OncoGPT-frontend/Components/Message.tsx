import { DocumentData } from "firebase/firestore"

type Props={
    message:DocumentData;
}

const Message = ({message}:Props) =>{
    const isGPT = message.user.name == "SpaceGPT";
    return (
        <div className="items-center flex justify-center mt-3">
        <div className={`py-5 text-white ${!isGPT && "bg-[#434654]"} rounded-xl max-w-[700px] w-full p-5`}>
        <div className="flex space-x-5 pt-5 max-w-2xl mx-auto">
            <img src={message.user.avatar} alt="" className="h-8 w-8 rounded-full"></img>
            <p className="pt-1 text-sm">
                {message.text}
            </p>
        </div>
        </div>
        </div>
    )
}

export default Message;