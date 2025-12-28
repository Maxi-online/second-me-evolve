// https://eslint.org/docs/latest/rules/
module.exports = {
  extends: ['eslint:recommended'],
  rules: {
    'array-callback-return': 'error',
    'no-console': [
      'warn',
      {
        allow: ['info', 'warn', 'error']
      }
    ],
    'no-else-return': [
      'warn',
      {
        allowElseIf: false
      }
    ],
    'no-empty': [
      'error',
      {
        allowEmptyCatch: true
      }
    ],
    'no-implicit-coercion': [
      'warn',
      {
        allow: ['!!', '+', '~']
      }
    ],
    'no-param-reassign': [
      'warn',
      {
        props: true,
        ignorePropertyModificationsFor: ['event', 'e']
      }
    ],
    'no-nested-ternary': 'off',
    'no-new': 'warn',
    'no-unused-vars': [
      'error',
      {
        ignoreRestSiblings: true,
        argsIgnorePattern: '^-',
        destructuredArrayIgnorePattern: '^_'
      }
    ],
    'no-shadow': [
      'error',
      {
        builtinGlobals: true,
        hoist: 'all'
      }
    ],
    'padding-line-between-statements': [
      'warn',
      {
        blankLine: 'always',
        next: '*',
        prev: ['const', 'let', 'var', 'if', 'for', 'while', 'switch', 'try']
      },
      {
        blankLine: 'any',
        next: ['const', 'let', 'var'],
        prev: ['const', 'let', 'var']
      },
      {
        blankLine: 'always',
        next: ['return', 'throw', 'break', 'continue', 'if', 'for', 'while', 'switch', 'try'],
        prev: '*'
      }
    ],
    quotes: ['warn', 'single']
  }
};
