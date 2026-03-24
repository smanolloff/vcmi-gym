import logging
import signal
import sqlite3
import sys
import time
import os
import re
import requests
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, List

DB_PATH = "autorent.db"
SLEEP_SECONDS = 60

# 9950X with 15MB/s dload speed inits in ~10 minutes
INIT_TIMEOUT_MINUTES = 30

VASTAI_API_KEY = os.environ["VASTAI_API_KEY"]
VASTAI_ENV = dict(
    AWS_ACCESS_KEY=os.environ["VASTAI_AWS_ACCESS_KEY"],
    AWS_SECRET_KEY=os.environ["VASTAI_AWS_SECRET_KEY"],
    VCMI_ARCHIVE_KEY=os.environ["VASTAI_VCMI_ARCHIVE_KEY"],
    WANDB_API_KEY=os.environ["VASTAI_WANDB_API_KEY"],
    VAST_API_KEY=os.environ["VASTAI_BENCHMARK_API_KEY"]
)

BADCPU_PATTERN = re.compile("EPYC|Xeon", re.IGNORECASE)

# LOG_LEVEL = logging.DEBUG
LOG_LEVEL = logging.INFO


def setup_logging() -> None:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )


def vastai_get_user():
    logging.info("vastai_get_user()")
    headers = {"Authorization": f"Bearer {VASTAI_API_KEY}"}
    url = "https://console.vast.ai/api/v0/users/current/"

    response = requests.get(url, headers=headers)
    logging.info(f"GET {url} {response.status_code}")
    logging.debug(f"Response body: {response.text}")

    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return response.json()


def vastai_search(blacklist: List[int]):
    logging.info(f"vastai_search([{','.join(str(x) for x in blacklist)}])")
    headers = {"Authorization": f"Bearer {VASTAI_API_KEY}"}

    body = {
        "type": "on-demand",
        "order": [["dph_total", "asc"]],
        "allocated_storage": 25.0,
        "external": {"eq": True},
        "rented": {"eq": False},
        "gpu_name": {"eq": "RTX 5090"},
        "cpu_cores": {"gte": "16"},
        "cpu_ram": {"gte": 55000.0},
        "dph_total": {"lte": "0.15"},       # <--- $/hr
        "rentable": {"eq": True},
    }

    if blacklist:
        body["machine_id"] = dict(notin=[str(x) for x in blacklist])

    url = "https://console.vast.ai/api/v0/bundles/"

    logging.debug(f"Request body: {body}")
    response = requests.post(url, headers=headers, json=body)
    logging.info(f"POST {url} {response.status_code}")
    logging.debug(f"Response body: {response.text}")

    assert response.status_code == 200, f"{response.status_code} {response.text}"
    data = response.json()

    return [
        offer for offer in data["offers"]
        if not BADCPU_PATTERN.search(offer["cpu_name"])
    ]


def vastai_get(instance_id):
    logging.info(f"vastai_list_instances({instance_id})")
    url = f"https://console.vast.ai/api/v0/instances/{instance_id}/?owner=me"
    headers = {"Authorization": f"Bearer {VASTAI_API_KEY}"}
    response = requests.get(url, headers=headers)
    logging.info(f"GET {url} {response.status_code}")
    logging.debug(f"Response body: {response.text}")
    assert response.status_code == 200, str(response.status_code)
    data = response.json()
    assert isinstance(data["instances"], dict), data["instances"].__class__
    return data["instances"]


def vastai_list():
    logging.info("vastai_list()")
    url = "https://console.vast.ai/api/v0/instances/?owner=me"
    headers = {"Authorization": f"Bearer {VASTAI_API_KEY}"}
    response = requests.get(url, headers=headers)
    logging.info(f"GET {url} {response.status_code}")
    logging.debug(f"Response body: {response.text}")
    assert response.status_code == 200, str(response.status_code)
    data = response.json()
    return {instance["id"]: instance for instance in data["instances"]}


def vastai_rent(offer_id: int) -> int:
    logging.info(f"vastai_rent({offer_id})")
    url = f"https://console.vast.ai/api/v0/asks/{offer_id}/"
    headers = {"Authorization": f"Bearer {VASTAI_API_KEY}"}
    body = dict(
        client_id="me",
        env=VASTAI_ENV,
        disk=25.0,
        template_hash_id="9535ff4084fd850b4c1cae890febf5e0",
        label="autorent",
        # :v
        onstart=(
            'set -x; touch ~/.no_auto_tmux; mkdir -p /workspace; cd /workspace;'
            'curl -sLO https://raw.githubusercontent.com/smanolloff/vcmi-gym/refs/heads/main/misc/vastai/preinit.sh;'
            'source preinit.sh; set +e; env -u FAKETIME_SHARED; unset FAKETIME_SHARED;'
            r'tmux new-session -d unset\ FAKETIME_SHARED\;'
            r'bash\ -xc\ unset\\\ FAKETIME_SHARED\\\;'
            r'cd\\\ /workspace\\\;'
            r'bash\\\ init.sh\\\;'
            r'bash\\\ check.sh\\\ -t\\\ -i90\\\ -r28\\\ -n5\;'
            r'exec\ \$SHELL'
        )
    )

    logging.debug(f"Request body: {body}")
    response = requests.put(url, headers=headers, json=body)
    logging.info(f"PUT {url} {response.status_code}")
    logging.debug(f"Response body: {response.text}")

    if response.status_code == 200:
        data = response.json()
        assert data["success"]
        return data["new_contract"]
    else:
        return None


def vastai_destroy(instance_id: int) -> None:
    logging.info(f"vastai_destroy({instance_id})")
    url = f"https://console.vast.ai/api/v0/instances/{instance_id}/"
    headers = {"Authorization": f"Bearer {VASTAI_API_KEY}"}

    response = requests.delete(url, headers=headers)
    logging.info(f"DELETE {url} {response.status_code}")
    logging.debug(f"Response body: {response.text}")

    assert response.status_code == 200, str(response.status_code)


@contextmanager
def db_connection(path: str):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    try:
        yield conn
    finally:
        conn.close()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS instances (
            instance_id INT PRIMARY KEY,
            machine_id INT NOT NULL,
            host_id INT NOT NULL,
            status TEXT,
            created_at DATETIME NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS blacklist (
            machine_id INT NOT NULL,
            host_id INT NOT NULL,
            counter INT NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            PRIMARY KEY (machine_id, host_id)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS warnlist (
            machine_id INT NOT NULL,
            host_id INT NOT NULL,
            counter INT NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            PRIMARY KEY (machine_id, host_id)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS goldlist (
            machine_id INT NOT NULL,
            host_id INT NOT NULL,
            counter INT NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            PRIMARY KEY (machine_id, host_id)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY,
            message TEXT,
            created_at DATETIME NOT NULL
        )
        """
    )

    conn.commit()


def db_blacklist_get(conn: sqlite3.Connection):
    # "Forget" blacklist after 7 days (to prevent random failures)
    sql = """
        SELECT machine_id
        FROM blacklist
        WHERE updated_at > datetime('now', '-7 days')
        """

    logging.debug(f"SQL: {sql}")
    rows = conn.execute(sql).fetchall()
    return [row["machine_id"] for row in rows]


def db_blacklist_add(conn: sqlite3.Connection, machine_id: int, host_id: int):
    logging.info(f"db_blacklist_add({machine_id}, {host_id})")
    sql = f"""
        INSERT INTO blacklist (machine_id, host_id, counter, created_at, updated_at)
        VALUES ({machine_id}, {host_id}, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT(machine_id, host_id)
        DO UPDATE SET counter = blacklist.counter + 1, updated_at = CURRENT_TIMESTAMP
        """
    logging.debug(f"SQL: {sql}")
    conn.execute(sql)
    conn.commit()


def db_warnlist_add(conn: sqlite3.Connection, machine_id: int, host_id: int):
    logging.info(f"db_warnlist_add({machine_id}, {host_id})")
    sql = f"""
        INSERT INTO warnlist (machine_id, host_id, counter, created_at, updated_at)
        VALUES ({machine_id}, {host_id}, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT(machine_id, host_id)
        DO UPDATE SET counter = blacklist.counter + 1, updated_at = CURRENT_TIMESTAMP
        """
    logging.debug(f"SQL: {sql}")
    conn.execute(sql)
    conn.commit()


def db_warnlist_del(conn: sqlite3.Connection, machine_id: int, host_id: int):
    logging.info(f"db_warnlist_del({machine_id}, {host_id})")
    sql = f"DELETE FROM warnlist WHERE machine_id = {machine_id} AND host_id = {host_id}"
    logging.debug(f"SQL: {sql}")
    conn.execute(sql)
    conn.commit()


def db_goldlist_add(conn: sqlite3.Connection, machine_id: int, host_id: int):
    logging.info(f"db_goldlist_add({machine_id}, {host_id})")
    sql = f"""
        INSERT INTO goldlist (machine_id, host_id, counter, created_at, updated_at)
        VALUES ({machine_id}, {host_id}, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ON CONFLICT(machine_id, host_id)
        DO UPDATE SET counter = goldlist.counter + 1, updated_at = CURRENT_TIMESTAMP
        """
    logging.debug(f"SQL: {sql}")
    conn.execute(sql)
    conn.commit()


def db_warnlist_del_sel(conn: sqlite3.Connection, counter: int) -> None:
    sql = f"""
        DELETE FROM warnlist
        WHERE counter > {counter}
        RETURNING machine_id, host_id
        """
    logging.debug(f"SQL: {sql}")
    rows = conn.execute(sql).fetchall()
    conn.commit()
    return rows


def db_instances_get_pending(conn: sqlite3.Connection):
    sql = """
        SELECT instance_id, machine_id, host_id, status, created_at
        FROM instances
        WHERE status IS NULL
        """
    logging.debug(f"SQL: {sql}")
    return conn.execute(sql).fetchall()


def db_instance_update(conn: sqlite3.Connection, instance_id: int, status: str) -> None:
    logging.info(f"db_instance_update({instance_id}, '{status}')")
    sql = f"UPDATE instances SET status = '{status}' WHERE instance_id = {instance_id}"
    logging.debug(f"SQL: {sql}")
    conn.execute(sql)
    conn.commit()


def db_instance_add(conn: sqlite3.Connection, instance_id: int, machine_id: int, host_id: int) -> None:
    logging.info(f"db_instance_add({instance_id}, {machine_id}, {host_id})")
    sql = f"""
        INSERT INTO instances (instance_id, machine_id, host_id, status, created_at)
        VALUES ({instance_id}, {machine_id}, {host_id}, NULL, CURRENT_TIMESTAMP)
        """
    logging.debug(f"SQL: {sql}")
    conn.execute(sql)
    conn.commit()


def db_audit_log(conn: sqlite3.Connection, message: str) -> None:
    conn.execute(
        """
        INSERT INTO audit_logs (message, created_at)
        VALUES (?, CURRENT_TIMESTAMP)
        """,
        (message,)
    )
    conn.commit()


def parse_sqlite_timestamp(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def is_older_than_minutes(ts: str, minutes: int) -> bool:
    created_at = parse_sqlite_timestamp(ts)
    now = datetime.now(timezone.utc)
    return now - created_at > timedelta(minutes=minutes)


def migrate_warn_to_blacklist(conn: sqlite3.Connection) -> None:
    rows = db_warnlist_del_sel(conn, 3)
    for row in rows:
        db_blacklist_add(conn, row["machine_id"], row["host_id"])


def handle_pending_instances(conn: sqlite3.Connection, running_instances: Dict[int, dict]) -> None:
    running_ids = vastai_list()

    for row in running_instances:
        instance_id = row["instance_id"]
        machine_id = row["machine_id"]
        host_id = row["host_id"]
        created_at = row["created_at"]

        if instance_id not in running_instances:
            db_instance_update(conn, instance_id, "unknown")
            continue

        label = running_ids[instance_id]["label"]

        if label in ["autorent", "init...", "check..."] and is_older_than_minutes(created_at, INIT_TIMEOUT_MINUTES):
            vastai_destroy(instance_id)
            db_audit_log(conn, f"destroy: instance_id={instance_id} reason=timeout")
            db_instance_update(conn, instance_id, "timeout")
            db_blacklist_add(conn, machine_id, host_id)
        elif label == "FAILED":
            vastai_destroy(instance_id)
            db_audit_log(conn, f"destroy: instance_id={instance_id} reason=FAILED")
            db_instance_update(conn, instance_id, "FAILED")
            db_blacklist_add(conn, machine_id, host_id)
        elif label == "PASSED":
            # db_audit_log(conn, f"destroy: instance_id={instance_id} reason=PASSED")
            db_instance_update(conn, instance_id, "PASSED")
            db_goldlist_add(conn, machine_id, host_id)


def handle_new_offers(conn: sqlite3.Connection) -> None:
    blacklist = db_blacklist_get(conn)
    offers = vastai_search(blacklist)

    for offer in offers:
        n_pending = len(db_instances_get_pending(conn))
        if n_pending > 0:
            logging.info(f"Will not rent {offer['id']} (already have {n_pending} instances)")
            continue

        instance_id = vastai_rent(offer["id"])
        if instance_id:
            instance = vastai_get(instance_id)
            db_audit_log(conn, f"rent: offer_id={offer['id']} instance_id={instance_id}")
            db_instance_add(conn, instance_id, instance["machine_id"], instance["host_id"])
        else:
            db_warnlist_add(conn, offer["machine_id"], offer["host_id"])


def main_loop() -> None:
    with db_connection(DB_PATH) as conn:
        ensure_schema(conn)

    while True:
        try:
            credit = vastai_get_user()["credit"]
            running_instances = vastai_list()
            n_instances = len(running_instances)
            if credit < 10 * n_instances:
                logging.info(f"Sleeping 600 seconds due to low balance (credit=${credit:.2f} n_instances={n_instances})")
                time.sleep(600)
                continue

            with db_connection(DB_PATH) as conn:
                migrate_warn_to_blacklist(conn)
                handle_pending_instances(conn, running_instances)
                handle_new_offers(conn)
        except Exception:
            logging.exception("Loop iteration failed")

        logging.info(f"Sleeping {SLEEP_SECONDS} seconds")
        time.sleep(SLEEP_SECONDS)


def install_signal_handlers() -> None:
    def _handle_signal(signum, frame):
        logging.info("Received signal %s, exiting", signum)
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)


if __name__ == "__main__":
    setup_logging()
    install_signal_handlers()
    main_loop()


# VASTAI IDs:
#
# 1. GET offers (/api/v0/bundles/)
# Relevant fields for 1 offer:
# {
#   "id": 32493628,                 // only for rent; lost afterwards
#   "bundle_id": 1886923819,
#   "cpu_cores": 16,
#   "cpu_cores_effective": 16.0,
#   "cpu_name": "AMD Ryzen 7 7700 8-Core Processor",
#   "duration": 1760530.6279759407,
#   "host_id": 307092,
#   "machine_id": 43973,
#   "instance": {
#     "totalHour": 0.020833333333333332,
#   }
# }
#
# 2. RENT 32493628
# {"success": true, "new_contract": 33402296, "instance_api_key": "e8620fcad067a82e2223917522665911c209d8656d8bfb8060a927ca8788fc1f"}
#
# 3. GET instances
# Relevant fields for 1 instance:
# {
#   "host_id": 307092,
#   "id": 33402296,
#   "instance": {
#     "totalHour": 0.020833333333333332,
#   },
#   "machine_dir_ssh_port": 51199,  // ??
#   "machine_id": 43973,
#   "ports": {
#     "22/tcp": [
#       {
#         "HostIp": "0.0.0.0",
#         "HostPort": "30022"
#       },
#     ]
#   },
#   "public_ipaddr": "139.59.37.138",
#   "ssh_host": "ssh3.vast.ai",
#   "ssh_port": 12296,
# }

# $ vastai destroy instance 33402296
# destroying instance 33402296.
