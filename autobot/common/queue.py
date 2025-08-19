from frozendict import frozendict
from queue import Queue


class MessageBroker:
    def __init__(self, topics: list[str]):
        self.topics = frozendict({topic: Queue() for topic in topics})
    
    def __validate_topic(self, topic: str):
        if topic not in self.topics:
            raise Exception(f"Topic={topic} does not exist")
    
    def put(self, topic: str, msg: any):
        self.__validate_topic(topic)
        self.topics[topic].put(msg)

    def get(self, topic: str):
        self.__validate_topic(topic)
        return self.topics[topic].get()
    
    def empty(self, topic: str):
        self.__validate_topic(topic)
        return self.topics[topic].empty()
