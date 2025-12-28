// https://typescript-eslint.io/rules/
module.exports = {
  extends: ['plugin:@typescript-eslint/recommended'],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    project: ['./tsconfig.json'],
    createDefaultProgram: true
  },
  overrides: [
    {
      files: ['*.ts', '*.tsx'],
      rules: {
        '@typescript-eslint/explicit-module-boundary-types': 'warn'
      }
    }
  ],
  rules: {
    '@typescript-eslint/array-type': [
      'warn',
      {
        default: 'array'
      }
    ],
    'comma-dangle': 'off',
    '@typescript-eslint/comma-dangle': 'warn',
    '@typescript-eslint/consistent-type-exports': 'warn',
    '@typescript-eslint/consistent-type-imports': 'warn',
    '@typescript-eslint/explicit-function-return-type': 'off',
    '@typescript-eslint/explicit-module-boundary-types': 'off',
    'no-empty-function': 'off',
    '@typescript-eslint/no-empty-function': 'warn',
    '@typescript-eslint/no-empty-interface': [
      'warn',
      {
        allowSingleExtends: true
      }
    ],
    '@typescript-eslint/no-explicit-any': 'warn',
    'no-loss-of-precision': 'off',
    '@typescript-eslint/no-loss-of-precision': 'error',
    'no-redeclare': 'off',
    '@typescript-eslint/no-redeclare': 'error',
    '@typescript-eslint/no-unnecessary-condition': 'warn',
    'no-unused-vars': 'off',
    '@typescript-eslint/no-unused-vars': [
      'error',
      {
        ignoreRestSiblings: true,
        argsIgnorePattern: '^_',
        destructuredArrayIgnorePattern: '^_'
      }
    ],
    'no-shadow': 'off',
    '@typescript-eslint/no-shadow': [
      'error',
      {
        // builtinGlobals: true,
        hoist: 'all',
        ignoreTypeValueShadow: false,
        ignoreFunctionTypeParameterNameValueShadow: false
      }
    ],
    '@typescript-eslint/non-nullable-type-assertion-style': 'warn',
    '@typescript-eslint/prefer-enum-initializers': 'warn',
    '@typescript-eslint/prefer-literal-enum-member': 'warn',
    '@typescript-eslint/prefer-optional-chain': 'warn',
    '@typescript-eslint/prefer-ts-expect-error': 'warn',
    quotes: 'off',
    '@typescript-eslint/quotes': ['warn', 'single']
  }
};
