# Stranger Outing Recommender – Vision & System Design

> **Working name:** Roam-style Agentic Social Outing Recommender
> **Owner:** KR x OP
> **Scope (this doc):** Problem framing, goals, and architecture for the **recommendation engine** that powers both **place** and **people** suggestions in a stranger-meetup app.

---

## 1. Vision & Problem Statement

### 1.1 What we are building

We are building a **social outing platform** where:

1. Users discover **places** (entertainment, sports, clubs, dining, etc.).
2. Users discover **compatible strangers** to go with.
3. The system eventually **automates logistics** (agentic bookings / reservations) once a group agrees.

The core experience is:

* *“I’m in X mood, at Y time, near Z area → show me a place + 2–4 people I’d actually enjoy going with.”*

To support this, we need a **recommendation engine** that simultaneously understands:

* A user’s **activity & venue preferences**.
* A user’s **social / vibe compatibility** with other people on the platform.
* **Spatial/temporal constraints** (where they are, how far they can go, when they’re free).
* **Incentives to attend** (likelihood of them actually showing up if invited).

### 1.2 Why this is non-trivial

This is not a plain item recommender (like “users who bought X also bought Y”). We need to:

1. Recommend **places** (items) *and* **people** (other users).
2. Do **spatial analysis**: choose venues that make sense geographically and logistically.
3. Do **social compatibility**: match strangers who likely vibe well, not just share a random interest.
4. Provide **explanations**: “You both like fishing”, “You all frequent live music spots in Brooklyn” – to build trust and help as conversation starters.

This pushes us towards a **graph-based, multi-objective recommendation system**, not just a simple collaborative filter.

---

## 2. Desired Product Outcomes

### 2.1 High-level outcomes

1. **Place Recommendations (Spatial Analysis)**

   * For a given user (or a small group) and time window, propose **top-N venues** that:

     * Match their **stated interests** and **implicit behavior**.
     * Are **geographically reasonable** (travel time / area constraints).
     * Have appropriate **vibe / category** (entertainment, sports, clubs, dining, etc.).

2. **People Recommendations (Social Compatibility)**

   * For a given user and (optionally) a planned activity/place, propose **top-N candidate strangers** who:

     * Share relevant **interest tags** / content consumption patterns.
     * Are **nearby enough** to make plans practical.
     * Have a high **compatibility score** – i.e., strong chance of accepting and actually attending.

3. **Explainability & Trust**

   * Every recommendation (place or person) should come with a simple, human-readable **reason**:

     * “You both like fishing.”
     * “You both go to late-night dining spots in Manhattan.”
     * “Matches your interest in board-game bars.”

4. **Performance & UX**

   * Recommendations must feel **instant** (sub-second response for typical queries) even as the user base and place inventory scale up.

---

## 3. Data & Signals We Will Use

We assume the following signals are available or can be collected.

### 3.1 Places & Tags

* **Place categories (coarse)**:

  * entertainment, sports, clubs, dining, etc. (5–8 broad tags).
* **Place tags (fine-grained)**:

  * Derived from posts and metadata about the place. Examples: fishing, bouldering, techno, live music, board games, rooftop, brunch, karaoke.
* **Location attributes**:

  * Latitude/longitude, city, neighborhood, typical visitors’ home areas, typical time windows (evenings, weekends).
* **Operational attributes**:

  * Price band, open hours, capacity constraints (if available).

### 3.2 User Behavior & Interests

1. **Post-level interactions**

   * Time spent viewing posts (dwell time).
   * Likes, saves, shares, comments.
   * Each post is already tagged with **fine-grained tags**, and linked to a place.

2. **Real-time & historical location**

   * Coarse-grained current location (city / neighborhood level, not precise tracking).
   * History of areas frequently visited.

3. **Expressed interests (self-reported)**

   * Users can explicitly select interest tags they like.
   * We may maintain a **top-10 dynamic tag profile** per user, updated as they post and interact.

4. **Social signals**

   * Soft indicators that two users are related:

     * Co-attendance at the same place/time.
     * Heavy overlap in highly weighted interests.
     * In-app interactions (messages, group participation, etc.) once available.

### 3.3 Derived Signals & Features

From the raw data, we will derive:

* **User broad preference vector** over coarse categories (entertainment/sports/clubs/dining/...).
* **User fine interest vector** for detailed tags (fishing, techno, etc.), with weights from dwell time + interactions.
* **Place category and tag vectors** derived from posts and metadata.
* **Soft social edges** between users based on co-attendance and interest overlap.
* **Implicit ratings** from behavior (mapped dwell time & action types to a 1–5 scale).

---

## 4. Core Recommendation Objectives (Engine Requirements)

The recommendation engine must satisfy the requirements framed in the task description.

### 4.1 Spatial Analysis (Best Location)

For a given user (or group) and context (time window, mood, constraints), we must:

1. **Score candidate places** using:

   * Preference match: user-place interest/category alignment.
   * Spatial compatibility: distance / travel time constraints.
   * Temporal fit: open hours vs desired time, user’s past behavior at similar hours.
   * Popularity / conversion priors: places with high “show-up” rates.
   * Novelty / diversity: avoid recommending the same place over and over.

2. **Return top-N places** with scores and explanations.

### 4.2 Social Compatibility (Best People)

For a given user (and optional candidate place/activity), we must:

1. **Compute compatibility scores** with other users:

   * Shared interests and tag overlaps.
   * Overlapping place categories and activity styles.
   * Location proximity and overlapping times.
   * Historical evidence of co-attendance or similar patterns.

2. **Return top-N users** ranked by compatibility score.

3. Provide **incentive-aware scores** (likelihood to accept & attend):

   * Based on how often they commit/attend to similar invitations historically.

### 4.3 Explainability

We need a consistent way to answer:

* *Why did we recommend this place?*
  → Because it strongly matches your tags (music + clubs), is 10 minutes away, and you usually go out at this time.

* *Why did we recommend this person?*
  → Because you both like fishing and late-night dining, and you both frequent the same neighborhoods.

Explainability must be:

* Derived directly from the features used in the model (no fake reasons).
* Expressible in a short, user-friendly phrase.

### 4.4 Performance & Scalability

* Must support **fast queries**: retrieving place and person recommendations in real time.
* Must scale to:

  * Large numbers of users and places.
  * Frequent updates as behavior data flows in.
* Must be amenable to **incremental improvements** (new features, new model variants) without rewriting everything.

---

## 5. Conceptual Modeling Strategy

We choose a **graph-based representation** of our world and a **graph neural network (GNN)** approach (GraphRec-style) for learning.

### 5.1 Why graphs?

Our domain is naturally graph-shaped:

* Users connect to places (interactions).
* Users connect to tags (interests).
* Places connect to tags.
* Users connect to other users (soft social ties).

Graphs allow us to:

* Capture **higher-order relationships** (users linked via shared places/tags).
* Combine **social, behavioral, and content signals** in a single model.
* Learn **embeddings** that represent both users and places in the same vector space.

### 5.2 Single Backbone, Multiple Heads

We will use **one shared GNN backbone** that produces:

* A **user embedding** for every user.
* A **place embedding** for every place.

On top of this backbone, we will attach **two task-specific heads**:

1. **Place Recommendation Head (Spatial Analysis)**

   * Input: user embedding, place embedding, contextual features (distance, time slot).
   * Output: scalar score = suitability of place for user (or group).

2. **Friend Compatibility Head (Social Compatibility)**

   * Input: user embedding (u), user embedding (v), contextual features (distance, planned activity).
   * Output: scalar score = compatibility for going out together.

This gives us:

* A unified representation of behavior & preferences.
* Clean separation between **place ranking** and **friend ranking** logic.
* Shared learning: improvements in user embeddings help both tasks.

### 5.3 Compatibility Scores vs Recommendations

At the model level, we always compute **scores**:

* `compat_friend(u, v | context)`
* `score_place(u, p | context)`

The **recommendations** are simply:

* Top-K places with maximum `score_place` (after filtering).
* Top-K users with maximum `compat_friend` (after filtering & constraints).

This means we inherently support both:

* Raw **compatibility scoring APIs** (for orchestration and group formation logic).
* High-level **recommendation APIs** (for app UIs).

---

## 6. Chosen Methodology: GraphRec-Style GNN

### 6.1 Core idea of GraphRec

GraphRec is a family of models that:

1. Take a **user–item interaction graph** with edge opinions (e.g., ratings).
2. Take a **user–user social graph** with edge strengths.
3. Learn user and item embeddings via message passing over both graphs.
4. Use these embeddings to predict user–item preference scores.

We adapt this idea to our domain:

* **Users** are users.
* **Items** are **places**.
* **Social graph** is our **soft user–user edges** (constructed from co-attendance & interests).

### 6.2 How we adapt GraphRec to our needs

1. **Graph structure**

   * Nodes:

     * User nodes.
     * Place nodes.
   * Edges:

     * User–Place edges with implicit ratings.
     * User–User edges with edge weights.
   * Optionally, we can later add Tag nodes for more expressiveness.

2. **Backbone**

   * User embeddings are updated based on:

     * Their own features (broad prefs, interest tags, home area).
     * Aggregated messages from neighboring user nodes (social ties).
     * Aggregated messages from neighboring place nodes (interactions and opinions).
   * Place embeddings are updated based on:

     * Their features (tags, category, price, location).
     * Aggregated messages from users who interacted with them.

3. **Heads**

   * Place head: predict `score_place(u, p | ctx)` for ranking venues.
   * Friend head: predict `compat_friend(u, v | ctx)` for ranking people.

4. **Training objectives**

   * Place task:

     * Contrast positive interactions (strong signals like attend/saves) against negatives (unseen places) using pairwise or classification loss.
   * Friend task:

     * Use historical co-attendance, accepted invitations, or successful group events as positives.
     * Sample random user pairs as negatives.

This results in embeddings and heads tailored for **our dual-objective problem**: place selection and friend matching.

---

## 7. System Architecture (High-Level)

### 7.1 Components Overview

1. **Data Ingestion & Storage**

   * Store users, places, posts, interactions, and location data.

2. **Feature Engineering & Graph Builder**

   * Periodic jobs that aggregate data into user features, place features, and edges.

3. **Model Training Pipeline**

   * Offline training jobs that build the GraphRec-like model with two heads.

4. **Embedding Export & Indexing**

   * Export learned user/place embeddings.
   * Build fast approximate-nearest-neighbor (ANN) indices for retrieval.

5. **Online Recommendation Service**

   * REST/gRPC APIs that, given a user and optional context, return:

     * Top places (spatial analysis).
     * Top people (social compatibility).
     * Optional raw compatibility scores.

6. **Explanation Service**

   * Lightweight layer that, given a recommendation, extracts top shared features and converts them into human-readable reasons.

7. **(Later) Agent Orchestrator**

   * Once a group is formed and a place is chosen, trigger **agentic workflows** for reservations/bookings.

### 7.2 Data Flow (End-to-End)

1. **User uses app** → generates events: views posts, likes, saves, attends events, moves around.
2. **Events stored** in interaction and location tables.
3. **Batch jobs** aggregate events into user/place profiles and build soft social edges.
4. **Graph builder** translates DB contents into tensors/structures for training.
5. **GNN training job** optimizes embeddings and task heads.
6. **Embeddings exported** and loaded into ANN indices.
7. **Online request** (user opens the app / pings "find something to do"):

   * Fetch user embedding.
   * Retrieve candidate places and people via ANN.
   * Score with heads, filter & rank.
   * Generate explanations.
   * Return to client.

---

## 8. Stepwise Plan to Achieve This

### 8.1 Phase 0 – Clarify Product Scenarios

* Define 3–5 canonical user flows, e.g.:

  1. **Solo place discovery**: user opens app, gets top 10 places nearby.
  2. **Solo + people**: user wants a group for a specific activity (e.g., bowling tonight).
  3. **Group extension**: existing pair wants 2 more for a board-game bar.
* For each flow, specify:

  * Inputs (user id, location, time, desired tags, etc.).
  * Outputs (places, people, explanations).

### 8.2 Phase 1 – Baseline Recommender (No GNN yet)

Before going full GraphRec, build a simple baseline to de-risk:

* User & place vectors built from tag frequencies and simple location features.
* Cosine similarity for:

  * user → place (content-based filter) for spatial analysis.
  * user → user (common tags) for friend suggestions.
* Naive explanations from overlapping tags.

This provides:

* A baseline product that works.
* First dataset of “what users actually click/join” to later supervise the GNN.

### 8.3 Phase 2 – GraphRec Backbone for Places

* Implement a GraphRec-like model focused on **user–place + user–user** graphs.
* Train only the **place recommendation head** initially.
* Deploy it as the engine for place suggestions.
* Compare against baseline (click-through, join rates).

### 8.4 Phase 3 – Add Friend Compatibility Head

* Define labels for friend compatibility:

  * co-attendance to same event.
  * accepted invites / joined groups.
* Train the **friend head** on top of the same backbone.
* Expose APIs for recommending people.

### 8.5 Phase 4 – Tighten Spatial & Incentive Modeling

* Add richer spatial features (travel time, distance bands) into heads.
* Model acceptance probability and show-up probability explicitly.
* Consider 2-stage ranking: fast candidate retrieval → heavy re-ranker.

### 8.6 Phase 5 – Agent Orchestration Integration

* Once the recommendation engine is stable, integrate with:

  * Reservation APIs (OpenTable, booking platforms, etc.).
  * Payment or ticket purchase flows.
* Design policies where, after a group agrees, an agent:

  * Confirms times.
  * Books the place.
  * Sends confirmations and reminders.

---

## 9. Risks, Assumptions & Guardrails

### 9.1 Risks

1. **Cold start**: new users and new places have little data.
2. **Data sparsity**: stranger meetups might be rare initially.
3. **Privacy & trust**: users may be uncomfortable with highly personal inferences.
4. **Overfitting**: GNN might learn patterns that don’t generalize.

### 9.2 Mitigations

* Use **content features** (tags, categories) heavily, not only interaction signals.
* Start with interpretable, simple baselines, then layer complexity.
* Aggregate and coarsen location data (neighborhood/city level) to protect privacy.
* Use conservative explanations that don’t reveal sensitive behavior.
* Proper offline validation + online A/B tests.

### 9.3 Key Assumptions

* We can collect enough user behavior data to train a meaningful model.
* Users are willing to state some initial interests (cold-start questionnaire).
* We have enough venue coverage in each geography to give non-trivial suggestions.

---

## 10. Summary

We are designing a **graph-based, dual-purpose recommendation engine** for a social outing app focused on:

* **Spatial Analysis**: finding the right venue for a user or group, given preferences and constraints.
* **Social Compatibility**: finding the right strangers to go with, based on interests, behavior, and location.

We will:

1. Represent the ecosystem as a **user–place–user graph** enriched with tags and soft social ties.
2. Use a **GraphRec-style GNN backbone** to learn user and place embeddings.
3. Attach two heads:

   * a **place head** for venue scoring.
   * a **friend head** for people compatibility scoring.
4. Serve recommendations via **fast vector search + re-ranking**, with human-readable explanations.
5. Gradually evolve from a simple baseline to a sophisticated, agent-integrated system.

This document is the conceptual and architectural foundation. Next steps are LLD for modules, schemas, and training pipelines, which we can refine incrementally on top of this base.
