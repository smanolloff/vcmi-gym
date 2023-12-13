import threading
import time

# Condition variable
condition = threading.Condition()


# Producer function to add items to the shared queue
def producer():
    while True:
        time.sleep(0.1)
        with condition:
            print("PRODUCER")
            condition.notify()
            condition.wait()


# Consumer function to consume items from the shared queue
def consumer():
    while True:
        with condition:
            # Wait for the producer to notify about the availability of items
            condition.wait()
            print("CONSUMER")
            condition.notify()


# Creating and starting producer and consumer threads
producer_thread = threading.Thread(target=producer)
consumer_thread = threading.Thread(target=consumer)

producer_thread.start()
consumer_thread.start()
