#include <iostream>
#include <vector>
#include <string>
#include <memory>

// Template class for a generic queue
template<typename T>
class Queue {
private:
    struct Node {
        T data;
        std::unique_ptr<Node> next;
        Node(T value) : data(value), next(nullptr) {}
    };
    
    std::unique_ptr<Node> front;
    Node* back;
    size_t size;

public:
    Queue() : front(nullptr), back(nullptr), size(0) {}
    
    void enqueue(T value) {
        auto newNode = std::make_unique<Node>(value);
        if (isEmpty()) {
            front = std::move(newNode);
            back = front.get();
        } else {
            back->next = std::move(newNode);
            back = back->next.get();
        }
        size++;
    }
    
    T dequeue() {
        if (isEmpty()) {
            throw std::runtime_error("Queue is empty");
        }
        T value = front->data;
        front = std::move(front->next);
        if (front == nullptr) {
            back = nullptr;
        }
        size--;
        return value;
    }
    
    bool isEmpty() const {
        return front == nullptr;
    }
    
    size_t getSize() const {
        return size;
    }
};

int main() {
    // Test with integers
    Queue<int> intQueue;
    intQueue.enqueue(1);
    intQueue.enqueue(2);
    intQueue.enqueue(3);
    
    std::cout << "Integer Queue:" << std::endl;
    while (!intQueue.isEmpty()) {
        std::cout << intQueue.dequeue() << " ";
    }
    std::cout << std::endl;
    
    // Test with strings
    Queue<std::string> strQueue;
    strQueue.enqueue("Hello");
    strQueue.enqueue("World");
    strQueue.enqueue("!");
    
    std::cout << "\nString Queue:" << std::endl;
    while (!strQueue.isEmpty()) {
        std::cout << strQueue.dequeue() << " ";
    }
    std::cout << std::endl;
    
    return 0;
} 