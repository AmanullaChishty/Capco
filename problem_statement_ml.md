# ML Engineer — Take-Home Assessment

## The Problem

You are building a **Record Matching System** for a small internal platform. The platform helps a sales team track client meetings by pulling data from two upstream systems. Records in these systems refer to the same real-world meetings, but there is no shared identifier between them.

Your job is to build a system that decides which records match, measure how well it works, and explain where it fails.

### Data Sources

**Source A — CRM API** (`/data/crm_events.json`)
Contains meeting records with client info, meeting dates, and relationship owners.

**Source B — Calendar API** (`/data/calendar_events.json`)
Contains calendar entries with attendees, times, locations, and recurrence info.

### Evaluation Labels

**Partial Ground Truth** (`/data/evaluation_labels.json`)
Contains labeled pairs — some confirmed matches, some confirmed non-matches. This is a subset. There are additional true matches in the data that are not labeled.

---

## The Data

Both data files are provided in `/data/`. You'll notice:

- Records across sources don't share a common ID
- Some meetings appear in both sources, some in only one
- Timestamps don't always match exactly
- Some fields conflict between sources (e.g., a meeting location in the calendar says "Zoom" but the CRM says "In-Person")
- One source has a record that looks like a duplicate of another record in the same source
- There are records with missing or malformed fields

We have intentionally not told you how to handle any of these cases.

---

## What You Need to Build

A system that:

1. **Ingests** data from both sources
2. **Decides** which records refer to the same real-world meeting
3. **Evaluates** how well those decisions hold up against the provided labels
4. **Serves** predictions through a simple API

---

## Requirements

**Functional:**

- A matching pipeline that produces match decisions with some measure of confidence
- Evaluation of your system against the provided labels
- A simple REST API that serves match predictions
- A README documenting your approach, your results, and your key decisions

**Non-Functional:**

- Use any language, framework, or tools you're comfortable with
- The pipeline should run with a single command (document it)
- Include a README

**That's it.** There is no further specification. The approach — rules, heuristics, similarity scoring, embeddings, a trained model — is entirely up to you. The data is small. Choose accordingly.

---

## What We're Evaluating

We are **not** evaluating:

- Use of a specific ML framework or technique
- Whether you trained a model (heuristics are a valid and potentially correct choice)
- Visual design or frontend polish
- Catching every edge case

We **are** evaluating:

1. **How you approach the matching problem** — What decisions do you make, what assumptions do you state, and how do you reason about trade-offs?
2. **How you know it works** — We want evidence, not just assertions. You have labels. Use them.
3. **How you handle bad data** — The data has problems. What you do about them matters more than catching every one.
4. **Your decision documentation** — For every meaningful decision, tell us what you chose, what you considered, and why.
5. **Code organization** — How you separate concerns across the pipeline.

---

## Submission

- A Git repository (or zip) with your code
- A README covering: how to run it, your approach, and your key decisions
- Time spent (be honest — we calibrate to this)

Expected time: **until due date**. If you find yourself going significantly over, stop and document what you would have done with more time.

---

## One More Thing

We expect you to use AI coding tools. We'd like you to include a short section in your README about how you used them and where, if anywhere, you found they needed correction or weren't helpful.