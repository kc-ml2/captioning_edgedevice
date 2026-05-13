func buildPrompt(question: String) -> String {
    
    let system = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    
    let prompt =
        system + " " +
        "USER: " + "<image>\n" + question + " " +
        "ASSISTANT:"
    
    return prompt
}
