############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.  The
# ASF licenses this file to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the
# License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations
# under the License.
#
############################################################################

import os
import shlex
import socket
import threading

import pytest

from ntfc.coreconfig import CoreConfig
from ntfc.device.getdev import get_device
from ntfc.device.renode import DeviceRenode


def _conf(**extra):
    cfg = {
        "name": "t",
        "device": "renode",
        "exec_args": '-e "include @config/renode/board.resc"',
        "boot_timeout": 2,
        "uptime": 0,
    }
    cfg.update(extra)
    conf = CoreConfig(cfg)
    conf._config["elf_path"] = "some/nuttx"
    return conf


class FakeChild:
    def __init__(self, alive=True, output=(), quits=True):
        self.quits = quits
        self.alive = alive
        self.output = list(output)
        self.sent = []
        self.pid = 4242

    def isalive(self):
        return self.alive

    def read_nonblocking(self, size=0, timeout=0):
        import pexpect

        if not self.output:
            raise pexpect.TIMEOUT("no data")
        return self.output.pop(0)

    def sendline(self, data):
        self.sent.append(data)
        if data == "quit" and self.quits:
            self.alive = False


class Listener:
    """Minimal TCP peer standing in for Renode's socket terminal."""

    def __init__(self):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.conn = None
        self.thread = threading.Thread(target=self._accept, daemon=True)
        self.thread.start()

    def _accept(self):
        self.conn, _ = self.srv.accept()

    def wait(self):
        self.thread.join(timeout=5)
        assert self.conn is not None
        return self.conn

    def close(self):
        self.conn.close()
        self.srv.close()


def test_device_renode_factory(envconfig_dummy):
    envconfig_dummy.product_get(0)["cores"]["core0"]["device"] = "renode"
    dev = get_device(envconfig_dummy.product[0].cfg_core(0))
    assert isinstance(dev, DeviceRenode)
    assert dev.name == "renode"


def test_device_renode_start_command():
    dev = DeviceRenode(_conf())
    cmds = []
    dev.host_open = lambda cmd, uptime: cmds.append((cmd, uptime))
    dev.start()

    cmd, uptime = cmds[0]
    assert uptime == 0
    argv = shlex.split("".join(cmd))

    assert argv[0] == "renode"
    assert argv[1:4] == ["--disable-gui", "--console", "--plain"]

    execs = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
    assert execs[0] == "$bin=@" + os.path.abspath("some/nuttx")
    assert execs[1] == (
        f'emulation CreateServerSocketTerminal {dev._port} "ntfc" false'
    )
    # user arguments run after the terminal exists, before start
    assert execs[2] == "include @config/renode/board.resc"
    assert execs[3] == "start"
    assert dev._port > 0


def test_device_renode_start_custom_exec_path():
    dev = DeviceRenode(_conf(exec_path="/opt/renode/renode"))
    cmds = []
    dev.host_open = lambda cmd, uptime: cmds.append(cmd)
    dev.start()
    assert shlex.split("".join(cmds[0]))[0] == "/opt/renode/renode"


def test_device_renode_start_requires_exec_args():
    dev = DeviceRenode(_conf(exec_args=""))
    with pytest.raises(KeyError):
        dev.start()


def test_device_renode_post_spawn_connects():
    listener = Listener()
    dev = DeviceRenode(_conf())
    dev._child = FakeChild()
    dev._port = listener.port

    assert dev._dev_is_health_priv() is False
    dev._post_spawn()
    listener.wait()
    assert dev._dev_is_health_priv() is True

    dev._stop_impl = lambda: None  # keep FakeChild, only exercise socket
    dev._close_sock()
    assert dev._dev_is_health_priv() is False
    listener.close()


def test_device_renode_post_spawn_timeout():
    # bound but not listening: connect fails until the deadline
    dev = DeviceRenode(_conf(boot_timeout=0))
    dev._child = FakeChild()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dev._port = s.getsockname()[1]
        with pytest.raises(TimeoutError):
            dev._post_spawn()


def test_device_renode_post_spawn_retries_until_listening():
    dev = DeviceRenode(_conf(boot_timeout=5))
    dev._child = FakeChild()
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    dev._port = srv.getsockname()[1]

    # not listening yet: the first connects are refused
    timer = threading.Timer(0.3, srv.listen, args=(1,))
    timer.start()
    dev._post_spawn()
    conn, _ = srv.accept()
    assert dev._dev_is_health_priv() is True

    dev._close_sock()
    conn.close()
    srv.close()


def test_device_renode_read_without_child():
    dev = DeviceRenode(_conf())
    assert dev._read() == b""


def test_device_renode_read_connection_reset():
    import struct

    listener = Listener()
    dev = DeviceRenode(_conf())
    dev._child = FakeChild()
    dev._port = listener.port
    dev._post_spawn()
    peer = listener.wait()

    # RST instead of FIN: recv raises and the console is dropped
    peer.setsockopt(
        socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
    )
    peer.close()
    import time

    time.sleep(0.05)
    assert dev._read() == b""
    assert dev._dev_is_health_priv() is False
    listener.srv.close()


def test_device_renode_post_spawn_child_died():
    dev = DeviceRenode(_conf(boot_timeout=5))
    dev._child = FakeChild(alive=False, output=[b"Renode error log"])
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dev._port = s.getsockname()[1]
        with pytest.raises(IOError):
            dev._post_spawn()


def test_device_renode_read_write():
    listener = Listener()
    dev = DeviceRenode(_conf())
    child = FakeChild(output=[b"renode log line"])
    dev._child = child
    dev._port = listener.port
    dev._post_spawn()
    peer = listener.wait()

    # nothing pending
    assert dev._read() == b""
    # renode's own output was drained from the child
    assert child.output == []

    peer.sendall(b"nsh> ")
    import time

    time.sleep(0.05)
    assert dev._read() == b"nsh> "

    dev._write(b"help")
    assert peer.recv(100) == b"help\n"
    dev._write(b"ls\n")
    assert peer.recv(100) == b"ls\n"

    dev._write_ctrl("c")
    assert peer.recv(100) == b"\x03"

    # peer closes: read returns nothing and the device turns unhealthy
    peer.close()
    time.sleep(0.05)
    assert dev._read() == b""
    assert dev._dev_is_health_priv() is False
    assert dev._read() == b""
    dev._write(b"ignored")
    dev._write_ctrl("c")
    listener.close()


def test_device_renode_stop_quits_monitor_and_kills():
    listener = Listener()
    dev = DeviceRenode(_conf())
    child = FakeChild()
    dev._child = child
    dev._port = listener.port
    dev._post_spawn()
    listener.wait()

    killed = []
    dev._kill_process_group = lambda proc: killed.append(proc)

    dev._stop_impl()
    assert child.sent == ["quit"]
    assert killed == []
    assert dev._child is None
    assert dev._sock is None
    listener.close()

    # a monitor that ignores quit is killed
    dev.QUIT_TIMEOUT = 0.2
    dev._child = FakeChild(quits=False)
    dev._stop_impl()
    assert len(killed) == 1
    assert dev._child is None

    # stop on a stopped device is a no-op
    dev._stop_impl()


def test_device_renode_reopen_picks_new_port():
    dev = DeviceRenode(_conf())
    ports = []

    def fake_host_open(cmd, uptime=0):
        ports.append(dev._port)
        dev._child = FakeChild()
        return dev._child

    dev.host_open = fake_host_open
    dev.start()
    dev._dev_reopen()
    assert len(ports) == 2
    assert all(p > 0 for p in ports)
    # the old process is gone before the new one starts, so a fresh
    # ephemeral port is drawn each time rather than reusing the command
    assert dev._child is not None
