// Class representing a simple stack data structure
class Stack {
    constructor() {
        this.items = [];
    }

    push(element) {
        this.items.push(element);
    }

    pop() {
        if (this.isEmpty()) {
            return "Stack is empty";
        }
        return this.items.pop();
    }

    peek() {
        if (this.isEmpty()) {
            return "Stack is empty";
        }
        return this.items[this.items.length - 1];
    }

    isEmpty() {
        return this.items.length === 0;
    }

    size() {
        return this.items.length;
    }
}

// Example usage
function main() {
    const stack = new Stack();
    
    // Push some elements
    stack.push(10);
    stack.push(20);
    stack.push(30);
    
    console.log("Stack size:", stack.size());
    console.log("Top element:", stack.peek());
    
    // Pop elements
    console.log("Popped:", stack.pop());
    console.log("Popped:", stack.pop());
    console.log("Popped:", stack.pop());
    
    console.log("Is stack empty?", stack.isEmpty());
}

main(); 