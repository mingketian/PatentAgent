# Product QA Checklist

This file is for manually testing the **product mode**:

- `Our Patent Agent`
- `Normal RAG baseline`
- `Our optimized patent agent`

Use it to check:
- vague patent search
- claim-style search
- similar patent search
- follow-up handling
- aspect filtering
- combination exploration
- insufficiency behavior

## 1. How to use this checklist

Recommended testing flow:

1. run a query in `Normal RAG baseline`
2. run the same query in `Our optimized patent agent`
3. compare:
   - retrieved patents
   - evidence quality
   - answer usefulness
   - follow-up handling

If needed, also test:
- `Show optional evidence snippets for baseline view`

## 2. What to look for

### In `Normal RAG baseline`

Expected:
- top patents only
- optional evidence snippets
- no full agent answer
- no planner-based context reuse

### In `Our optimized patent agent`

Expected:
- grounded answer
- supporting evidence
- planner / action summary
- verification status
- follow-up questions should use prior context when appropriate

## 3. Core 20 test queries

### A. Vague search

**Q1**
```text
patents about event participants getting personalized information during a live event
```

**Q2**
```text
patents related to attendee profiles and contextual information sharing at physical gatherings
```

**Q3**
```text
find patents about participant data, event context, and customized attendee experience
```

**Q4**
```text
patents for people at an event requesting information about other participants
```

What to check:
- do the top patents look on-topic?
- does optimized mode produce a more useful explanation than baseline?

---

### B. Claim-style retrieval

**Q5**
```text
1. A method for leveraging social networks in physical gatherings, the method comprising: generating a profile for each participant; receiving participant data; receiving a request for information from a participant; determining whether the participant has access rights; identifying responsive information; and providing the identified information.
```

**Q6**
```text
1. A computer-implemented method comprising: retrieving participant attributes from one or more data sources for an event; receiving an inquiry during the event for contextual information about another participant; determining whether the requester is authorized to receive the information; and returning the contextual information to the requester.
```

**Q7**
```text
1. A method of engaging event attendees comprising: providing attendee-specific content to attendee devices, customizing activities for attendees, and updating interactions based on attendee preferences and event activity data.
```

What to check:
- are the retrieved patents technically close?
- does optimized mode explain relevance better?

---

### C. Similar patent search

**Q8**
```text
find patents similar to systems and methods for presenting information extracted from one or more data sources to event participants
```

**Q9**
```text
find patents similar to an invention about participant profiles, access control, and event information delivery
```

**Q10**
```text
what patents are closest to a system that gives attendees contextual information derived from multiple data sources
```

What to check:
- are retrieved patents still anchored in the same technical space?
- does optimized mode make the similarity judgment clearer?

---

### D. Aspect filtering over previous results

Run one event-related query first, then ask these as follow-ups.

**Q11**
```text
Which of those also includes access control for the requested information?
```

**Q12**
```text
Which retrieved patent is strongest on participant profile generation?
```

**Q13**
```text
Which one seems more focused on attendee customization than on information access rights?
```

**Q14**
```text
Which of the retrieved patents explicitly mentions contextual information requests?
```

What to check:
- optimized mode should reuse the working set
- baseline should not have the same follow-up capability

---

### E. Comparison questions

Ask these after a relevant initial retrieval.

**Q15**
```text
Compare the top two patents for relevance to participant profile handling and contextual information delivery.
```

**Q16**
```text
Which is the stronger match for event participant context: the first or second retrieved patent?
```

**Q17**
```text
What is the main difference between the top two retrieved patents?
```

What to check:
- optimized mode should produce an actual comparison
- baseline should not behave like a full comparison agent

---

### F. Combination exploration

**Q18**
```text
If I combine participant profile-based access control with smart invitations, what related patents should I inspect next?
```

**Q19**
```text
I want to combine event participant contextual information with invitation management. What related patents are closest?
```

What to check:
- does optimized mode broaden the search in a sensible way?
- does it avoid pretending the combination is already fully supported if evidence is weak?

---

### G. Out-of-domain / insufficiency

**Q20**
```text
patents about quantum error correction for superconducting qubits
```

What to check:
- does optimized mode admit weak evidence or low confidence?
- does it avoid overconfident unrelated claims?

## 4. Best short test pack

If you only have time for a quick demo QA pass, use these:

1. `Q1`
2. `Q5`
3. `Q11`
4. `Q15`
5. `Q18`
6. `Q20`

## 5. Pass/fail notes

### Product baseline passes if:
- it returns plausible top patents
- optional snippets work
- it stays simple

### Optimized agent passes if:
- it returns a grounded answer
- evidence aligns with the answer
- follow-up queries reuse context
- verification is displayed

### Warning signs

Flag these if they happen:
- baseline looks too similar to the optimized agent
- follow-up queries in optimized mode ignore previous results
- evidence snippets do not support the answer
- out-of-domain queries still get overconfident answers
