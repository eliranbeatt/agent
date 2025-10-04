# UI Integration Tests

This directory contains integration tests for the Local Agent Studio UI.

## Current Status

The test files contain placeholder test structures that document the expected behavior and test scenarios. To run these tests in a production environment, you'll need to:

## Setup Instructions

### 1. Install Testing Dependencies

```bash
npm install --save-dev @testing-library/react @testing-library/jest-dom @testing-library/user-event jest jest-environment-jsdom
```

### 2. Configure Jest

Create `jest.config.js`:

```javascript
const nextJest = require('next/jest')

const createJestConfig = nextJest({
  dir: './',
})

const customJestConfig = {
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  testEnvironment: 'jest-environment-jsdom',
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
}

module.exports = createJestConfig(customJestConfig)
```

### 3. Create Jest Setup File

Create `jest.setup.js`:

```javascript
import '@testing-library/jest-dom'
```

### 4. Update package.json

Add test script:

```json
{
  "scripts": {
    "test": "jest",
    "test:watch": "jest --watch"
  }
}
```

## Test Coverage

The test suite covers:

### File Upload and Processing
- Drag and drop functionality
- File format validation
- Upload progress tracking
- Error handling

### Chat Interface
- Message display and history
- Real-time streaming responses
- Keyboard shortcuts
- Input validation

### Execution Monitoring
- Execution mode toggling
- Real-time status updates
- Resource usage tracking
- Agent monitoring
- Plan graph visualization

### Execution Controls
- Pause/Resume functionality
- Stop execution
- State management

### Source Panel
- Source display and navigation
- Citation tracking
- Metadata display

### Accessibility
- Keyboard navigation
- ARIA labels
- Responsive design

## Running Tests

Once dependencies are installed:

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm test -- --coverage
```

## Implementation Notes

The current test files serve as documentation and specification for the expected UI behavior. Each test case describes:
- The user action being tested
- The expected system response
- Any edge cases or error conditions

To implement these tests, you would:
1. Set up the testing environment as described above
2. Replace placeholder test bodies with actual test implementations
3. Mock API calls to the FastAPI backend
4. Mock WebSocket connections for streaming
5. Use React Testing Library to render components and simulate user interactions
