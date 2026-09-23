#include "nexusdata/backend/thread_pool.hpp"

namespace nexusdata {

ThreadPool& default_thread_pool() {
    static ThreadPool pool;
    return pool;
}

} // namespace nexusdata
