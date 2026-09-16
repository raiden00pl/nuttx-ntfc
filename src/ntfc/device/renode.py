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

"""Renode device: a spawned emulator with the console on a TCP socket."""

import os
import select
import shlex
import socket
import time
from typing import TYPE_CHECKING, Optional

import pexpect  # type: ignore

from ntfc.log.logger import logger

from .host import DeviceHost

if TYPE_CHECKING:
    from ntfc.coreconfig import CoreConfig

###############################################################################
# Class: DeviceRenode
###############################################################################


class DeviceRenode(DeviceHost):
    """Host-based Renode emulator.

    Renode runs headless with its monitor on the child pty. NTFC creates a
    raw TCP socket terminal named ``ntfc`` that the target script connects
    its console UART to. Process ownership comes from :class:`DeviceHost`.
    """

    DEFAULT_EXEC = "renode"
    RENODE_FLAGS = ("--disable-gui", "--console", "--plain")
    TERMINAL_NAME = "ntfc"
    CONNECT_POLL = 0.1
    SOCK_TIMEOUT = 5.0
    QUIT_TIMEOUT = 3.0

    def __init__(self, conf: "CoreConfig"):
        """Initialize Renode device.

        :param conf: configuration handler
        """
        DeviceHost.__init__(self, conf)
        self._sock: Optional[socket.socket] = None
        self._port = 0

    @property
    def name(self) -> str:
        """Get device name."""
        return "renode"

    @staticmethod
    def _free_port() -> int:
        """Draw an ephemeral localhost port for the socket terminal."""
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def _start_impl(self) -> None:
        """Start Renode.

        NTFC sets ``$bin`` to the ELF and creates the socket terminal
        ``ntfc`` before the user arguments run; the script given there is
        expected to load ``$bin`` and connect its console UART to ``ntfc``.
        """
        elf = os.path.abspath(self._image_path())
        exec_args = self._conf.exec_args
        if not exec_args:
            raise KeyError("no exec_args in configuration file!")

        self._port = self._free_port()
        terminal = (
            f'CreateServerSocketTerminal {self._port} "{self.TERMINAL_NAME}"'
        )

        cmd = [self._conf.exec_path or self.DEFAULT_EXEC]
        cmd.append(" " + " ".join(self.RENODE_FLAGS))
        cmd.append(" -e " + shlex.quote(f"$bin=@{elf}"))
        cmd.append(" -e " + shlex.quote(f"emulation {terminal} false"))
        cmd.append(" " + exec_args)
        cmd.append(" -e start")

        self.host_open(cmd, self._conf.uptime)

    def _dev_reopen(self) -> pexpect.spawn:
        """Restart Renode on a fresh socket terminal port."""
        self._stop_impl()
        self._start_impl()
        assert self._child
        return self._child

    def _drain_child(self) -> None:
        """Forward Renode's own monitor/log output to the debug log."""
        if not self._child:
            return
        try:
            while True:
                out = self._child.read_nonblocking(size=5120, timeout=0)
                logger.debug(f"renode: {out!r}")
        except (pexpect.TIMEOUT, pexpect.EOF):
            pass

    def _post_spawn(self) -> None:
        """Connect to the socket terminal once Renode listens on it."""
        deadline = time.monotonic() + self._conf.boot_timeout
        while True:
            try:
                sock = socket.create_connection(
                    ("127.0.0.1", self._port), timeout=self.SOCK_TIMEOUT
                )
                break
            except OSError:
                self._drain_child()
                if not self._child or not self._child.isalive():
                    raise IOError("renode exited before opening console")
                if time.monotonic() >= deadline:
                    raise TimeoutError("renode console connect timeout")
                time.sleep(self.CONNECT_POLL)

        sock.settimeout(self.SOCK_TIMEOUT)
        self._sock = sock
        logger.info(f"renode console connected on port {self._port}")

    def _close_sock(self) -> None:
        """Close the console socket."""
        if self._sock:
            self._sock.close()
        self._sock = None

    def _dev_is_health_priv(self) -> bool:
        """Check if Renode runs and the console is connected."""
        return DeviceHost._dev_is_health_priv(self) and self._sock is not None

    def _read(self) -> bytes:
        """Read console data from the socket terminal."""
        self._drain_child()
        if not self.dev_is_health():
            return b""

        assert self._sock
        ready, _, _ = select.select([self._sock], [], [], 0)
        if not ready:
            return b""

        try:
            data = self._sock.recv(5120)
        except OSError:
            data = b""

        if not data:
            # peer closed the console
            self._close_sock()
        return data

    def _write(self, data: bytes) -> None:
        """Write to the console. One write per command: TCP is a stream."""
        if not self.dev_is_health():
            return

        if data[-1] != ord("\n"):
            data += self.NEWLINE_PAD

        assert self._sock
        self._sock.sendall(data)

    def _write_ctrl(self, c: str) -> None:
        """Write a control character to the console."""
        if not self.dev_is_health():
            return

        assert self._sock
        self._sock.sendall(bytes([ord(c.upper()) & 0x1F]))

    def _stop_impl(self) -> None:
        """Close the console, ask the monitor to quit, then kill if needed."""
        self._close_sock()

        child = self._child
        if child and child.isalive():
            child.sendline("quit")
            deadline = time.monotonic() + self.QUIT_TIMEOUT
            while child.isalive() and time.monotonic() < deadline:
                time.sleep(self.CONNECT_POLL)

        if child and child.isalive():
            self._kill_process_group(child)

        self._child = None
        logger.info("renode device closed")
