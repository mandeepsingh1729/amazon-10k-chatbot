import warnings
warnings.filterwarnings('ignore')

from chatbot import AmazonChatBot

def main():
    """Command-line interface for the chatbot"""
    print("=" * 50)
    print("🤖 Amazon 10-K ChatBot - CLI Version")
    print("=" * 50)
    print("Type 'quit' to exit, 'clear' to clear history, 'help' for commands\n")
    
    try:
        chatbot = AmazonChatBot()
        
        while True:
            question = input("\nYou: ").strip()
            
            if question.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'clear':
                chatbot.clear_history()
                print(" Chat history cleared")
                continue
            
            if question.lower() == 'help':
                print("\n Available commands:")
                print("  'quit'   - Exit the chatbot")
                print("  'clear'  - Clear chat history")
                print("  'help'   - Show this help")
                continue
            
            if not question:
                continue
            
            print("...")
            try:
                result = chatbot.answer(question)
                print(f"\n Bot: {result['answer']}")
                if result['sources']:
                    print(f"Sources: Pages {', '.join(result['sources'])}")
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again or check your API key.")
    
    except Exception as e:
        print(f" Failed to initialize chatbot: {e}")
        print("Make sure you've run data_loader.py and vector_store.py first!")

if __name__ == "__main__":
    main()