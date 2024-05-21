# kernel_tools
Tools for kernel methods

use `export USE_KERNEL_TOOLS_JIT=1` to force use of cpp/cu in JIT mode for development.

use `export USE_KERNEL_TOOLS_JIT=0` or don't export this symbol to fall back to compiled versions.

use `export USE_KERNEL_TOOLS_DEBUG=0` or don't export this symbol to compile cxx and cuda in release mode.

use `export USE_KERNEL_TOOLS_DEBUG=1` to compile cxx and cuda with no optimization. This should compile faster.

use `export USE_KERNEL_TOOLS_VERBOSE=1` to compile cxx and cuda verbose during JIT compilation. This shows your standard compile errors with line numbers in the cpp and cuda files.

use `export USE_KERNEL_TOOLS_VERBOSE=0` or don't export to silence the JIT compilation log lines.