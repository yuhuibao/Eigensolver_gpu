#include <stddef.h>
#include <stdio.h>
/**
 * \return the original pointer incremented by a number of bytes
 */
void* inc_c_ptr(void *ptr, ptrdiff_t diff_in_bytes) {
  return ptr+diff_in_bytes;
}
