package query

import (
	"github.com/mlflow/mlflow/mlflow/go/pkg/query/lexer"
	"github.com/mlflow/mlflow/mlflow/go/pkg/query/parser"
	"github.com/mlflow/mlflow/mlflow/go/pkg/utils"
)

func ParseFilter(input *string) ([]*parser.ValidCompareExpr, error) {
	if utils.IsNilOrEmptyString(input) {
		return make([]*parser.ValidCompareExpr, 0), nil
	}

	tokens, err := lexer.Tokenize(input)
	if err != nil {
		return nil, err
	}

	ast, err := parser.Parse(tokens)
	if err != nil {
		return nil, err
	}

	validExpressions := make([]*parser.ValidCompareExpr, 0, len(ast.Exprs))
	for _, expr := range ast.Exprs {
		ve, err := parser.ValidateExpression(expr)
		if err != nil {
			return nil, err
		}
		validExpressions = append(validExpressions, ve)
	}

	return validExpressions, nil
}
