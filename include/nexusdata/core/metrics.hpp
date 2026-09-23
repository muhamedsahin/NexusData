#pragma once

#include <string>
#include <string_view>

namespace nexusdata {
namespace metrics {

/// User sink (Prometheus, a test recorder, ...). Not owned.
class Sink {
public:
    virtual ~Sink() = default;
    virtual void add(std::string_view name, double value) = 0;
};

void set_sink(Sink* sink);
[[nodiscard]] Sink* sink();

/// Always forwards to the sink when one is installed.
void publish(std::string_view name, double value);

/// Hot path. Compiled to nothing unless NEXUSDATA_METRICS_ENABLED is set,
/// so instrumented call sites stay free in a default build.
inline void add(std::string_view name, double value) {
#if defined(NEXUSDATA_METRICS_ENABLED)
    publish(name, value);
#else
    (void)name;
    (void)value;
#endif
}

[[nodiscard]] inline bool enabled() {
#if defined(NEXUSDATA_METRICS_ENABLED)
    return true;
#else
    return false;
#endif
}

} // namespace metrics
} // namespace nexusdata
