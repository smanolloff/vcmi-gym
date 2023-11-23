import multiprocessing

lock = multiprocessing.Lock()

def myfunc():
    with lock:
        # Your synchronized code here
        print("asd")
        pass

def main_func():
    # Spawns a new process running myfunc
    p = multiprocessing.Process(target=myfunc)
    p.start()
    p.join()

if __name__ == "__main__":
    main_func()
