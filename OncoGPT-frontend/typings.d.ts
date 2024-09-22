interface Message {
    text:String
    createdAt: admin.firestore.Timestamp
    user:{
        _id:String
        name:String
        avatar:String
    }
}