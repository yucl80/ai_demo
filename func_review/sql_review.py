prompt="""
As an experienced Java development expert, please review the following Java source code. During your review, pay special attention to the following points:

### 1. SQL Injection Security:
   - If SQL statements are entirely composed of static string concatenations without any dynamic user input, please do not consider this as an SQL injection vulnerability.
   - SQL injection risks may only exist when SQL statements include unprocessed user input.
   - If prepared statements (PreparedStatement) or parameterized queries are used, it can generally be considered safe.
   - Here is an example of a secure SQL query using a prepared statement:
     ```java
     String query = "SELECT * FROM users WHERE username = ?";
     PreparedStatement pstmt = connection.prepareStatement(query);
     pstmt.setString(1, username);
     ResultSet rs = pstmt.executeQuery();
     ```

### 2. Code Quality and Best Practices:
   - Does the code follow Java coding conventions and best practices?
   - Are there any code smells or anti-patterns?
   - How is the readability and maintainability of the code?

### 3. Performance Optimization:
   - Are there any potential performance issues?
   - What areas can be optimized for better performance?

### 4. Other Security Considerations:
   - Apart from SQL injection, are there any other types of security vulnerabilities or risks?
   - Is sensitive data handled correctly?

### 5. Functional Completeness:
   - Does the code fully implement the expected functionality?
   - Are all possible edge cases and exceptions handled?

### 6. Test Coverage:
   - Is there sufficient unit test coverage for the code?
   - Are tests considered for critical paths and boundary conditions?

### 7. Design Patterns and Architecture:
   - What design patterns are used? Are they appropriate?
   - Is the overall code architecture reasonable?

### 8. Documentation and Comments:
   - Is the code adequately commented and documented?
   - Are method and class names clear and self-explanatory?

### Guidelines for Assessing SQL Injection Risks:
1. Carefully examine how SQL statements are constructed.
2. Distinguish between static SQL and dynamic SQL.
3. Identify if prepared statements or ORM frameworks are used.
4. If you find potential SQL injection risks, please explain the reasons and specific locations in detail.
5. Provide improvement suggestions, such as using prepared statements or parameterized queries.
6. Emphasize the importance of input validation and filtering.

### Example of Proper SQL Handling:
```java
// Example of a secure SQL query using a prepared statement
String query = "SELECT * FROM users WHERE username = ?";
PreparedStatement pstmt = connection.prepareStatement(query);
pstmt.setString(1, username);
ResultSet rs = pstmt.executeQuery();

"""