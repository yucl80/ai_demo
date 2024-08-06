DEFAULT_NEBULAGRAPH_NL2CYPHER_PROMPT_TMPL = """
Generate NebulaGraph query from natural language.
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Schema:
---
{schema}
---
Note: NebulaGraph speaks a dialect of Cypher, comparing to standard Cypher:

1. it uses double equals sign for comparison: `==` rather than `=`
2. it needs explicit label specification when referring to node properties, i.e.
v is a variable of a node, and we know its label is Foo, v.`foo`.name is correct
while v.name is not.

For example, see this diff between standard and NebulaGraph Cypher dialect:
```diff
< MATCH (p:person)-[:directed]->(m:movie) WHERE m.name = 'The Godfather'
< RETURN p.name;
---
> MATCH (p:`person`)-[:directed]->(m:`movie`) WHERE m.`movie`.`name` == 'The Godfather'
> RETURN p.`person`.`name`;
```

Question: {query_str}

NebulaGraph Cypher dialect query:
"""


DEFAULT_KG_RESPONSE_ANSWER_PROMPT_TMPL = """
The original question is given below.
This question has been translated into a Graph Database query.
Both the Graph query and the response are given below.
Given the Graph Query response, synthesise a response to the original question.

Original question: {query_str}
Graph query: {kg_query_str}
Graph response: {kg_response_str}
Response: 
"""