import json
import socket
import time


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 50050))
    sock.settimeout(1.0)

    end = time.time() + 45
    remote_addr = None
    sent = False
    seen = []
    print("listening_udp_50050")

    while time.time() < end:
        try:
            payload, addr = sock.recvfrom(65535)
        except socket.timeout:
            continue
        remote_addr = addr
        try:
            msg = json.loads(payload.decode("utf-8"))
        except Exception:
            continue

        msg_type = msg.get("MsgType", "")
        platform_name = msg.get("PlatformName", "")
        if platform_name and platform_name not in seen and len(seen) < 12:
            seen.append(platform_name)
            print("seen", msg_type, platform_name)

        if msg_type == "TaskAck":
            print("TaskAck", json.dumps(msg, ensure_ascii=False))
        elif msg_type == "AttackResult":
            print("AttackResult", json.dumps(msg, ensure_ascii=False))

        if remote_addr and not sent:
            task = {
                "MsgType": "AssignTask",
                "PlatformName": "red_attack_1",
                "Task": "FIRE_AGM",
                "TargetName": "blue_ground_1",
                "TargetPosition": [25.1666667, 121.1666667, 0.0],
                "Weapon": "agm",
            }
            sock.sendto(json.dumps(task).encode("utf-8"), remote_addr)
            print("sent", json.dumps(task), "to", remote_addr)
            sent = True

    print("done", "seen=", seen, "sent=", sent)


if __name__ == "__main__":
    main()
