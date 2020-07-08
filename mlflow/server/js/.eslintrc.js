module.exports = {
  'extends': [
    'airbnb-base',
    'react-app',
    'prettier',
  ],
  'parser': 'babel-eslint',
  'plugins': [
    'prettier',
    'react',
  ],
  'parserOptions': {
    'sourceType': 'module',
    'ecmaVersion': 7,
    'ecmaFeatures': {
      'jsx': true,
    },
  },
  'env': {
    'es6': true,
    'browser': true,
    'jest': true,
  },
  'globals': {
    'GridStackUI': true,
    'jQuery': true,
    'MG': true,
  },
  'settings': {
    'import/resolver': {
      'webpack': {
        'config': {
          'resolve': {
            // eslint-disable-next-line no-process-env
            'modules': ['node_modules', process.env.NODE_MODULES_PATH],
            'extensions': ['.js', '.jsx', '.ts', '.tsx'],
          },
        },
      },
    },
    'react': {
      'version': 'detect',
    },
  },
  'rules': {
    "prettier/prettier": "error",
    'accessor-pairs': 2,
    'array-bracket-spacing': 2,
    'array-callback-return': 2,
    'arrow-body-style': "off",
    'arrow-parens': "off",
    'arrow-spacing': 2,
    'block-scoped-var': 2,
    'block-spacing': 2,
    'brace-style': [2, "1tbs", { "allowSingleLine": false }],
    'callback-return': 2,
    'camelcase': "off",
    'class-methods-use-this': 0,
    'comma-dangle': [2, "always-multiline"],
    'comma-spacing': 2,
    'comma-style': 2,
    'computed-property-spacing': 2,
    'consistent-return': 2,
    'consistent-this': 0,
    'constructor-super': 2,
    'curly': 2,
    'default-case': 2,
    'dot-location': 2,
    'dot-notation': "off",
    'eol-last': 2,
    'eqeqeq': 2,
    'func-call-spacing': 2,
    'func-names': 2,
    'func-style': 0,
    'generator-star-spacing': 2,
    'global-require': 2,
    'guard-for-in': 2,
    'handle-callback-err': 2,
    'id-blacklist': 2,
    'id-length': 0,
    'id-match': 2,
    'import/default': 0,
    'import/export': 2,
    'import/extensions': ['error', 'always', {
      'js': 'never',
      'jsx': 'never',
      'ts': 'never',
      'tsx': 'never',
    }],
    'import/first': 0,
    'import/max-dependencies': 0,
    'import/named': 0,
    'import/namespace': 2,
    'import/newline-after-import': 2,
    'import/no-absolute-path': 0,
    'import/no-amd': 2,
    'import/no-commonjs': 0,
    'import/no-deprecated': 1,
    'import/no-duplicates': 2,
    'import/no-dynamic-require': 0,
    'import/no-extraneous-dependencies': [2, { 'devDependencies': true }],
    'import/no-internal-modules': 0,
    'import/no-mutable-exports': 2,
    'import/no-named-as-default': 0,
    'import/no-named-as-default-member': 2,
    'import/no-namespace': 2,
    'import/no-nodejs-modules': 2,
    'import/no-restricted-paths': 2,
    'import/no-unassigned-import': 0,
    'import/no-unresolved': 0,
    'import/no-webpack-loader-syntax': 0,
    'import/order': 0,
    'import/prefer-default-export': 0,
    'import/unambiguous': 0,
    'indent': 0,
    'indent-legacy': 0,
    'init-declarations': 0,
    'jsx-quotes': "off",
    'key-spacing': 2,
    'keyword-spacing': 2,
    'linebreak-style': 2,
    'lines-around-comment': 0,
    'max-depth': [2, 4],
    'max-len': ['error', 100, 2, {
      'ignoreUrls': false,
      'ignoreComments': false,
      'ignoreRegExpLiterals': true,
      'ignoreStrings': false,
      'ignoreTemplateLiterals': false,
      'ignorePattern': "^import\\s.+\\sfrom\\s'.+';",  // ignore import statements
    }],
    'max-lines': [2, 2000],
    'max-nested-callbacks': 2,
    'max-params': [2, 12],
    'max-statements': 0,
    'max-statements-per-line': 2,
    'multiline-ternary': 0,
    'new-cap': 0,
    'new-parens': 2,
    'newline-after-var': 0,
    'newline-before-return': 0,
    'newline-per-chained-call': "off",
    'no-alert': 2,
    'no-array-constructor': 2,
    'no-bitwise': 2,
    'no-caller': 2,
    'no-case-declarations': 2,
    'no-catch-shadow': "off",
    'no-class-assign': 2,
    'no-cond-assign': 2,
    'no-confusing-arrow': 0,
    'no-console': 0,
    'no-const-assign': 2,
    'no-constant-condition': 2,
    'no-continue': 0,
    'no-control-regex': 2,
    'no-debugger': 2,
    'no-delete-var': 2,
    'no-div-regex': 2,
    'no-dupe-args': 2,
    'no-dupe-class-members': 2,
    'no-dupe-keys': 2,
    'no-duplicate-case': 2,
    'no-duplicate-imports': 2,
    'no-else-return': "off",
    'no-empty': 2,
    'no-empty-character-class': 2,
    'no-empty-function': 2,
    'no-empty-pattern': 2,
    'no-eq-null': 2,
    'no-eval': 2,
    'no-ex-assign': 2,
    'no-extend-native': 2,
    'no-extra-bind': 2,
    'no-extra-boolean-cast': 2,
    'no-extra-label': 2,
    'no-extra-parens': 0,
    'no-extra-semi': 2,
    'no-fallthrough': 2,
    'no-floating-decimal': 2,
    'no-func-assign': 2,
    'no-global-assign': 2,
    'no-implicit-coercion': 0,
    'no-implicit-globals': 2,
    'no-implied-eval': 2,
    'no-inline-comments': 0,
    'no-inner-declarations': 2,
    'no-invalid-regexp': 2,
    'no-invalid-this': 0,
    'no-irregular-whitespace': 2,
    'no-iterator': 2,
    'no-label-var': 2,
    'no-labels': 2,
    'no-lone-blocks': 2,
    'no-lonely-if': 2,
    'no-loop-func': 2,
    'no-magic-numbers': 0,
    'no-mixed-operators': 2,
    'no-mixed-requires': 0,
    'no-mixed-spaces-and-tabs': 2,
    'no-multi-spaces': ['error', { 'ignoreEOLComments': true }],
    'no-multi-str': 2,
    'no-multiple-empty-lines': 2,
    'no-negated-condition': 0,
    'no-nested-ternary': "off",
    'no-new': 0,
    'no-new-func': 2,
    'no-new-object': 2,
    'no-new-require': 2,
    'no-new-symbol': 2,
    'no-new-wrappers': 2,
    'no-obj-calls': 2,
    'no-octal': 2,
    'no-octal-escape': 2,
    'no-param-reassign': 2,
    'no-path-concat': 2,
    'no-plusplus': 0,
    'no-process-env': 0,
    'no-process-exit': 2,
    'no-proto': 2,
    'no-prototype-builtins': 0,
    'no-redeclare': 2,
    'no-regex-spaces': 2,
    'no-restricted-imports': 2,
    'no-restricted-modules': 2,
    'no-restricted-syntax': 0,
    'no-return-assign': 0,
    'no-script-url': 2,
    'no-self-assign': 2,
    'no-self-compare': 2,
    'no-sequences': 2,
    'no-shadow': 2,
    'no-shadow-restricted-names': 2,
    'no-sparse-arrays': 2,
    'no-sync': 2,
    'no-tabs': 2,
    'no-template-curly-in-string': 2,
    'no-ternary': 0,
    'no-this-before-super': 2,
    'no-throw-literal': 2,
    'no-trailing-spaces': 2,
    'no-undef': 2,
    'no-undef-init': "off",
    'no-undefined': 0,
    'no-underscore-dangle': 0,
    'no-unexpected-multiline': 2,
    'no-unmodified-loop-condition': 2,
    'no-unneeded-ternary': 2,
    'no-unreachable': 2,
    'no-unsafe-finally': 2,
    'no-unsafe-negation': 2,
    'no-unused-expressions': 2,
    'no-unused-labels': 2,
    'no-unused-vars': 2,
    'no-use-before-define': 0,
    'no-useless-call': 2,
    'no-useless-computed-key': 2,
    'no-useless-concat': 2,
    'no-useless-constructor': 2,
    'no-useless-escape': 2,
    'no-useless-rename': 2,
    'no-var': 2,
    'no-void': 2,
    'no-warning-comments': 0,
    'no-whitespace-before-property': 2,
    'no-with': 2,
    'object-curly-newline': 0,
    'object-curly-spacing': "off",
    'object-property-newline': 0,
    'object-shorthand': [2, 'methods'],
    'one-var': 0,
    'one-var-declaration-per-line': 2,
    'operator-assignment': 2,
    'operator-linebreak': 0,
    'padded-blocks': 0,
    'prefer-arrow-callback': 0,
    'prefer-const': 2,
    'prefer-reflect': 0,
    'prefer-rest-params': 2,
    'prefer-spread': 2,
    'prefer-template': 0,
    'prettier/prettier': 'error',
    'quote-props': 0,
    'quotes': "off",
    'radix': 2,
    'react/display-name': 0,
    'react/forbid-component-props': 0,
    'react/forbid-prop-types': 0,
    'react/jsx-boolean-value': 2,
    'react/jsx-closing-bracket-location': 0,
    'react/jsx-curly-spacing': 2,
    'react/jsx-equals-spacing': 2,
    'react/jsx-filename-extension': 0,
    'react/jsx-first-prop-new-line': 0,
    'react/jsx-handler-names': 0,
    'react/jsx-indent': 0,
    'react/jsx-indent-props': 0,
    'react/jsx-key': 0,
    'react/jsx-max-props-per-line': 0,
    'react/jsx-no-bind': [0, { 'ignoreRefs': true }],
    'react/jsx-no-comment-textnodes': 2,
    'react/jsx-no-duplicate-props': 2,
    'react/jsx-no-literals': 0,
    'react/jsx-no-target-blank': 2,
    'react/jsx-no-undef': 2,
    'react/jsx-pascal-case': 2,
    'react/jsx-sort-props': 0,
    'react/jsx-space-before-closing': 0,
    'react/jsx-uses-react': 2,
    'react/jsx-uses-vars': 2,
    'react/jsx-wrap-multilines': 0,
    'react/no-children-prop': 0,
    'react/no-danger': 2,
    'react/no-danger-with-children': 0,
    'react/no-deprecated': 2,
    'react/no-did-mount-set-state': 2,
    'react/no-did-update-set-state': 2,
    'react/no-direct-mutation-state': 2,
    'react/no-find-dom-node': 0,
    'react/no-is-mounted': 2,
    'react/no-multi-comp': 0,
    'react/no-render-return-value': 0,
    'react/no-set-state': 0,
    'react/no-string-refs': 0,
    'react/no-unescaped-entities': 0,
    'react/no-unknown-property': 2,
    'react/no-unused-prop-types': 0,
    'react/prefer-es6-class': 0,
    'react/prefer-stateless-function': 0,
    'react/prop-types': 2,
    'react/react-in-jsx-scope': 2,
    'react/require-optimization': 0,
    'react/require-render-return': 2,
    'react/self-closing-comp': 0,
    'react/sort-comp': 0,
    'react/sort-prop-types': 0,
    'react/style-prop-object': 2,
    'require-jsdoc': 0,
    'require-yield': 2,
    'rest-spread-spacing': 2,
    'semi': 2,
    'semi-spacing': 2,
    'sort-imports': 0,
    'sort-keys': 0,
    'sort-vars': 0,
    'space-before-blocks': 2,
    'space-before-function-paren': [2, {
      'anonymous': 'always',
      'named': 'never',
      'asyncArrow': 'always',
    }],
    'space-in-parens': 2,
    'space-infix-ops': 2,
    'space-unary-ops': 0,
    'spaced-comment': [2, 'always', { 'exceptions': ['/'], 'markers': ['/'] }],
    'strict': 2,
    'symbol-description': 2,
    'template-curly-spacing': 2,
    'unicode-bom': 2,
    'use-isnan': 2,
    'valid-jsdoc': 0,
    'valid-typeof': 2,
    'vars-on-top': 0,
    'wrap-iife': 2,
    'wrap-regex': 0,
    'yield-star-spacing': 2,
    'yoda': 2,
    'function-paren-newline': 'off',
    'complexity': ['error', 20],
    'no-multi-assign': 'off',
    'no-useless-return': 'off',
    'prefer-destructuring': 'off',
    'no-restricted-globals': [
      2,
      'addEventListener',
      'blur',
      'close',
      'closed',
      'confirm',
      'defaultStatus',
      'event',
      'external',
      'defaultstatus',
      'find',
      'focus',
      'frameElement',
      'frames',
      'innerHeight',
      'innerWidth',
      'length',
      'locationbar',
      'menubar',
      'moveBy',
      'moveTo',
      'onblur',
      'onerror',
      'onfocus',
      'onload',
      'onresize',
      'onunload',
      'open',
      'opener',
      'opera',
      'outerHeight',
      'outerWidth',
      'pageXOffset',
      'pageYOffset',
      'parent',
      'print',
      'removeEventListener',
      'resizeBy',
      'resizeTo',
      'screen',
      'screenLeft',
      'screenTop',
      'screenX',
      'screenY',
      'scroll',
      'scrollbars',
      'scrollBy',
      'scrollTo',
      'scrollX',
      'scrollY',
      'self',
      'status',
      'statusbar',
      'stop',
      'toolbar',
      'top'
    ],
  },
  'overrides': [
    {
      'files': ['*.test.js', '*-test.js', '*-test.jsx', 'test/**'],
      'plugins': [
        'chai-expect',
        'chai-friendly',
      ],
      'env': {
        'mocha': true,
      },
      'globals': {
        'sinon': true,
        'chai': true,
        'expect': true,
        'assert': true,
        'testSuite': true,
      },
      'rules': {
        'func-names': 0,
        'max-len': ['error', 100, 2, {
          'ignoreStrings': true,
          'ignoreTemplateLiterals': true,
        }],
        'max-lines': 0,
        'chai-expect/missing-assertion': 2,
        'no-unused-expressions': 0,
        'chai-friendly/no-unused-expressions': 2,
      }
    }
  ]
};
