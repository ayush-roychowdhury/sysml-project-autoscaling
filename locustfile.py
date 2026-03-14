"""
locustfile.py — Locust workload generator for DeathStarBench Social Network

Set workload profile via WORKLOAD env var:
    WORKLOAD=sequential locust -f locustfile.py --host=http://localhost:8080 --headless -u 50 -r 10 --run-time 300s
    WORKLOAD=fanout locust -f locustfile.py --host=http://localhost:8080 --headless -u 50 -r 10 --run-time 300s
    WORKLOAD=mixed locust -f locustfile.py --host=http://localhost:8080 --headless -u 50 -r 10 --run-time 300s

Prerequisites:
    pip install locust
    python3 scripts/init_social_graph.py --graph=socfb-Reed98
"""

from locust import HttpUser, task, between
import random
import string
import os

NUM_USERS = 962  # Reed98 social graph size
WORKLOAD = os.environ.get("WORKLOAD", "mixed")


def random_text(length=64):
    return ''.join(random.choices(string.ascii_lowercase + ' ', k=length))


class SocialNetworkUser(HttpUser):
    wait_time = between(0.5, 1.5)

    def on_start(self):
        """Register/login a user at the start of each Locust user session."""
        self.user_id = random.randint(1, NUM_USERS)
        self.username = f"username_{self.user_id}"

    @task
    def do_action(self):
        """Pick an action based on the workload profile."""
        r = random.random()

        if WORKLOAD == "sequential":
            # 50% read_user_timeline, 40% read_home_timeline, 10% compose
            if r < 0.50:
                self._read_user_timeline()
            elif r < 0.90:
                self._read_home_timeline()
            else:
                self._compose_post()

        elif WORKLOAD == "fanout":
            # 5% read_user_timeline, 5% read_home_timeline, 90% compose
            if r < 0.05:
                self._read_user_timeline()
            elif r < 0.10:
                self._read_home_timeline()
            else:
                self._compose_post()

        else:  # mixed (default)
            # 30% read_user_timeline, 30% read_home_timeline, 40% compose
            if r < 0.30:
                self._read_user_timeline()
            elif r < 0.60:
                self._read_home_timeline()
            else:
                self._compose_post()

    def _read_user_timeline(self):
        user_id = random.randint(1, NUM_USERS)
        self.client.get(
            "/wrk2-api/user-timeline/read",
            params={"user_id": str(user_id), "start": "0", "stop": "10"},
            name="/wrk2-api/user-timeline/read"
        )

    def _read_home_timeline(self):
        user_id = random.randint(1, NUM_USERS)
        self.client.get(
            "/wrk2-api/home-timeline/read",
            params={"user_id": str(user_id), "start": "0", "stop": "10"},
            name="/wrk2-api/home-timeline/read"
        )

    def _compose_post(self):
        user_id = random.randint(1, NUM_USERS)
        username = f"username_{user_id}"
        text = random_text()

        # Optionally mention other users (increases fan-out)
        if random.random() < 0.3:
            mentioned_id = random.randint(1, NUM_USERS)
            text += f" @username_{mentioned_id}"

        self.client.post(
            "/wrk2-api/post/compose",
            data={
                "username": username,
                "user_id": str(user_id),
                "text": text,
                "media_ids": "[]",
                "media_types": "[]",
                "post_type": "0"
            },
            name="/wrk2-api/post/compose"
        )
