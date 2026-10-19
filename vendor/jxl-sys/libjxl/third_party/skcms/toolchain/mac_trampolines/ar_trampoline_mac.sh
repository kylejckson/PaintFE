#!/bin/bash
# Copyright 2022 Google LLC
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# Based on
# https://skia.googlesource.com/skia/+/0d4fcf388a6f9318e2a54fa85fddf3396d521767/toolchain/mac_trampolines/ar_trampoline_mac.sh.
external/clang_mac/bin/llvm-ar $@
