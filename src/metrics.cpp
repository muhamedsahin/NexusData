#include "nexusdata/core/metrics.hpp"

#include <atomic>

namespace nexusdata {
namespace metrics {
namespace {

std::atomic<Sink*> g_sink{nullptr};

} // namespace

void set_sink(Sink* sink) { g_sink.store(sink, std::memory_order_release); }

Sink* sink() { return g_sink.load(std::memory_order_acquire); }

void publish(std::string_view name, double value) {
    if (Sink* s = sink()) s->add(name, value);
}

} // namespace metrics
} // namespace nexusdata
