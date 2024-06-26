import time
from threading import Lock, Thread

from tminterface.client import Client
from tminterface.interface import TMInterface

class SimStateClient(Client):
    """
    Client for a TMInterface instance.
    Its only job is to get the simulation state that is used by the gym env for reward computation.
    """

    def __init__(self):
        super().__init__()
        self.sim_state = None

    def on_run_step(self, iface, _time: int):
        self.sim_state = iface.get_simulation_state()

    def on_checkpoint_count_changed(self, iface, current: int, target: int):
        if current == target:
            iface.prevent_simulation_finish()
            iface.give_up()

            # Write the time when the simulation ended to a file
            with open("end_time.txt", "a") as f:
                f.write(time.strftime("%H:%M:%S") + "\n")


class ThreadedClient:
    """
    Allows to run the Client in a separate thread, so that the gym env can run in the main thread.
    """

    def __init__(self) -> None:
        self.iface = TMInterface()
        self.tmi_client = SimStateClient()
        self._client_thread = Thread(target=self.client_thread, daemon=True)
        self._lock = Lock()
        self.data = None
        self._client_thread.start()

    def client_thread(self):
        client = SimStateClient()
        print("ok")

        self.iface.register(client)
        while self.iface.running:
            time.sleep(0)
            self._lock.acquire()
            self.data = client.sim_state
            self._lock.release()


if __name__ == "__main__":
    simthread = ThreadedClient()
