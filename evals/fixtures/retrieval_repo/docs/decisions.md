# Architecture decision record

## Adopting the messaging bus

We evaluated two options for the internal notification path:

- **Option A — direct HTTP calls** from each service to each consumer. Simple,
  but every producer needs to know its consumers, and a slow consumer stalls
  the producer.
- **Option B — a durable messaging bus** with at-least-once delivery. Needs
  operational investment, but decouples producers from consumers.

**Decision: we adopt Option B, the durable messaging bus.**

The deciding factor was the coupling cost, not throughput: at our message
volume a direct HTTP fan-out would have worked fine, but every new consumer
would have required a producer-side change and a deploy, which we had already
paid for three times that quarter.

Accepted consequence: consumers must be idempotent, because delivery is
at-least-once. Exactly-once delivery is explicitly out of scope.
