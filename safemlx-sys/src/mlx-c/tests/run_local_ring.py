#!/usr/bin/env python3
"""Run the real native Ring conformance in two independent localhost processes."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument("--binary",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--timeout",type=float,default=180)
parser.add_argument("--ranks",type=int,default=2)
parser.add_argument("--cases",type=int,default=4)
parser.add_argument("--case-prefix",default="RING_CASE")
parser.add_argument("--success-prefix",default="RING_CONFORMANCE_PASS")
args=parser.parse_args()
if args.ranks < 2 or args.cases < 1:parser.error("positive cases and at least two ranks are required")
args.output.mkdir(parents=True,exist_ok=False)
# Reserve distinct loopback ports while composing the exact host file. Ring
# owns the listeners after these sockets close and both children are spawned.
reserved=[socket.socket(socket.AF_INET,socket.SOCK_STREAM) for _ in range(args.ranks)]
for value in reserved:value.bind(("127.0.0.1",0))
ports=[value.getsockname()[1] for value in reserved]
hostfile=args.output/"hosts.json"
hostfile.write_text(json.dumps([[f"127.0.0.1:{port}"] for port in ports])+"\n")
for value in reserved:value.close()
children=[]
started=time.monotonic()
try:
    for rank in range(args.ranks):
        env=os.environ.copy()
        env.update(MLX_HOSTFILE=str(hostfile.resolve()),MLX_RANK=str(rank),DEVICE="cpu")
        log=(args.output/f"rank-{rank}.log").open("w")
        child=subprocess.Popen([str(args.binary.resolve())],env=env,stdout=log,stderr=subprocess.STDOUT)
        children.append((rank,child,log))
    while any(child.poll() is None for _,child,_ in children):
        if any(child.poll() not in (None,0) for _,child,_ in children):
            raise RuntimeError("a Ring rank failed; see rank logs")
        if time.monotonic()-started>args.timeout:raise TimeoutError("local Ring fixture timed out")
        time.sleep(0.05)
    for rank,child,log in children:
        log.flush()
        text=(args.output/f"rank-{rank}.log").read_text()
        if child.returncode or f"{args.success_prefix} rank={rank}" not in text or text.count(f"{args.case_prefix} rank=")!=args.cases:
            raise RuntimeError(f"rank {rank} did not complete every selected real Ring case")
    (args.output/"result.json").write_text(json.dumps({"passed":True,"ranks":args.ranks,"cases_per_rank":args.cases,
        "hostfile":str(hostfile),"seconds":time.monotonic()-started,"binary":str(args.binary.resolve())},indent=2)+"\n")
finally:
    for _,child,_ in children:
        if child.poll() is None:child.terminate()
    for _,child,log in children:
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:child.kill();child.wait()
        log.close()
