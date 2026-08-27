/// Errors produced while constructing tokenizers or rendering chat templates.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Tokenizer metadata is malformed or inconsistent.
    #[error("invalid tokenizer metadata: {0}")]
    InvalidTokenizer(String),

    /// Reading tokenizer or template data failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// Chat-template metadata is malformed.
    #[error("invalid chat_template: {0}")]
    InvalidChatTemplate(String),

    /// A named template collection has no selectable default.
    #[error(
        "chat_template collection has no default template; available templates: {available:?}"
    )]
    AmbiguousChatTemplate {
        /// Names of the templates present in the collection.
        available: Vec<String>,
    },

    /// MiniJinja could not compile or render the template.
    #[error(transparent)]
    RenderTemplate(#[from] minijinja::Error),

    /// continue_final_message is set but the final message does not appear in the chat after   
    /// applying the chat template! This can happen if the chat template deletes portions of
    /// the final message. Please verify the chat template and final message in your chat to
    /// ensure they are compatible.
    #[error(
        "continue_final_message is set but the final message does not appear in the chat after applying the chat template!"
    )]
    FinalMsgNotInChat,

    /// The Hugging Face tokenizer could not encode input or be constructed.
    #[error(transparent)]
    Encode(#[from] tokenizers::tokenizer::Error),

    /// JSON serialization of template inputs failed.
    #[error(transparent)]
    Serialize(#[from] serde_json::Error),
}
