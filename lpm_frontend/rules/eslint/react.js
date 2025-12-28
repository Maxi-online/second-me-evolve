// https://github.com/jsx-eslint/eslint-plugin-react#list-of-supported-rules
module.exports = {
  extends: [
    'plugin:react/recommended',
    'plugin:react/jsx-runtime',
    'plugin:react-hooks/recommended'
  ],
  settings: {
    react: {
      version: 'detect'
    }
  },
  rules: {
    'react/destructuring-assignment': ['warn', 'always'],
    'react/jsx-filename-extension': [
      'error',
      {
        extensions: ['jsx', 'tsx']
      }
    ],
    'react/jsx-handler-names': [
      'warn',
      {
        checkLocalVariables: true
      }
    ],
    'react/jsx-no-useless-fragment': [
      'warn',
      {
        allowExpressions: true
      }
    ],
    'react/jsx-one-expression-per-line': [
      'warn',
      {
        allow: 'single-child'
      }
    ],
    'react/jsx-sort-props': [
      'warn',
      {
        // multiline: 'last'
        reservedFirst: true
      }
    ],
    'react/no-unstable-nested-components': 'error',
    'react/no-unused-class-component-methods': 'error',
    'react/self-closing-comp': 'warn'
  }
};
