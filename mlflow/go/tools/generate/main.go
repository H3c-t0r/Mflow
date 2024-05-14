package main

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/iancoleman/strcase"
	"github.com/mlflow/mlflow/mlflow/go/tools/generate/discovery"
)

func mkImportSpec(value string) *ast.ImportSpec {
	return &ast.ImportSpec{Path: &ast.BasicLit{Kind: token.STRING, Value: value}}
}

var importStatements = &ast.GenDecl{
	Tok: token.IMPORT,
	Specs: []ast.Spec{
		mkImportSpec(`"strings"`),
		mkImportSpec(`"github.com/gofiber/fiber/v2"`),
		mkImportSpec(`"github.com/mlflow/mlflow/mlflow/go/pkg/protos"`),
	},
}

func mkStarExpr(e ast.Expr) *ast.StarExpr {
	return &ast.StarExpr{
		X: e,
	}
}

func mkSelectorExpr(x, sel string) *ast.SelectorExpr {
	return &ast.SelectorExpr{X: ast.NewIdent(x), Sel: ast.NewIdent(sel)}
}

func mkNamedField(name string, typ ast.Expr) *ast.Field {
	return &ast.Field{
		Names: []*ast.Ident{ast.NewIdent(name)},
		Type:  typ,
	}
}

func mkField(typ ast.Expr) *ast.Field {
	return &ast.Field{
		Type: typ,
	}
}

func mkMethodInfoInputPointerType(methodInfo discovery.MethodInfo) *ast.StarExpr {
	return mkStarExpr(mkSelectorExpr(methodInfo.PackageName, methodInfo.Input))
}

// Generate a method declaration on an service interface
func mkServiceInterfaceMethod(methodInfo discovery.MethodInfo) *ast.Field {
	return &ast.Field{
		Names: []*ast.Ident{ast.NewIdent(strcase.ToCamel(methodInfo.Name))},
		Type: &ast.FuncType{
			Params: &ast.FieldList{
				List: []*ast.Field{
					mkNamedField("input", mkMethodInfoInputPointerType(methodInfo)),
				},
			},
			Results: &ast.FieldList{
				List: []*ast.Field{
					mkField(mkStarExpr(mkSelectorExpr(methodInfo.PackageName, methodInfo.Output))),
					mkField(mkStarExpr(ast.NewIdent("MlflowError"))),
				},
			},
		},
	}
}

// Generate a service interface declaration
func mkServiceInterfaceNode(serviceInfo discovery.ServiceInfo) *ast.GenDecl {
	// We add one method to validate any of the input structs
	methods := make([]*ast.Field, 1, len(serviceInfo.Methods)+1)

	methods[0] = &ast.Field{
		Names: []*ast.Ident{ast.NewIdent("Validate")},
		Type: &ast.FuncType{
			Params: &ast.FieldList{
				List: []*ast.Field{
					mkNamedField("input", &ast.InterfaceType{Methods: &ast.FieldList{}}),
				},
			},
			Results: &ast.FieldList{
				List: []*ast.Field{
					mkField(&ast.ArrayType{Elt: ast.NewIdent("string")}),
				},
			},
		},
	}

	for _, method := range serviceInfo.Methods {
		endpointName := fmt.Sprintf("%s_%s", serviceInfo.Name, method.Name)
		if _, ok := ImplementedEndpoints[endpointName]; ok {
			methods = append(methods, mkServiceInterfaceMethod(method))
		}
	}

	// Create an interface declaration
	return &ast.GenDecl{
		Tok: token.TYPE, // Specifies a type declaration
		Specs: []ast.Spec{
			&ast.TypeSpec{
				Name: ast.NewIdent(serviceInfo.Name), // Interface name
				Type: &ast.InterfaceType{
					Methods: &ast.FieldList{
						List: methods,
					},
				},
			},
		},
	}
}

func saveASTToFile(fset *token.FileSet, file *ast.File, addComment bool, outputPath string) error {
	// Create or truncate the output file
	outputFile, err := os.Create(outputPath)
	if err != nil {
		return err
	}
	defer outputFile.Close()

	// Use a bufio.Writer for buffered writing (optional)
	writer := bufio.NewWriter(outputFile)
	defer writer.Flush()

	if addComment {
		writer.WriteString("// Code generated by mlflow/go/tools/generate/main.go. DO NOT EDIT.\n\n")
	}

	// Write the generated code to the file
	return format.Node(writer, fset, file)
}

// fun(arg1, arg2, ...)
func mkCallExpr(fun ast.Expr, args ...ast.Expr) *ast.CallExpr {
	return &ast.CallExpr{
		Fun:  fun,
		Args: args,
	}
}

// Shorthand for creating &expr
func mkAmpExpr(expr ast.Expr) *ast.UnaryExpr {
	return &ast.UnaryExpr{
		Op: token.AND,
		X:  expr,
	}
}

// err != nil
var errNotEqualNil = &ast.BinaryExpr{
	X:  ast.NewIdent("err"),
	Op: token.NEQ,
	Y:  ast.NewIdent("nil"),
}

// return err
var returnErr = &ast.ReturnStmt{
	Results: []ast.Expr{ast.NewIdent("err")},
}

func mkBlockStmt(stmts ...ast.Stmt) *ast.BlockStmt {
	return &ast.BlockStmt{
		List: stmts,
	}
}

func mkIfStmt(init ast.Stmt, cond ast.Expr, body *ast.BlockStmt) *ast.IfStmt {
	return &ast.IfStmt{
		Init: init,
		Cond: cond,
		Body: body,
	}
}

func mkAssignStmt(lhs, rhs []ast.Expr) *ast.AssignStmt {
	return &ast.AssignStmt{
		Lhs: lhs,
		Tok: token.DEFINE,
		Rhs: rhs,
	}
}

func mkBinaryExpr(x ast.Expr, op token.Token, y ast.Expr) *ast.BinaryExpr {
	return &ast.BinaryExpr{
		X:  x,
		Op: op,
		Y:  y,
	}
}

func mkReturnStmt(results ...ast.Expr) *ast.ReturnStmt {
	return &ast.ReturnStmt{
		Results: results,
	}
}

func mkKeyValueExpr(key string, value ast.Expr) *ast.KeyValueExpr {
	return &ast.KeyValueExpr{
		Key:   ast.NewIdent(key),
		Value: value,
	}
}

func mkAppRoute(method discovery.MethodInfo, endpoint discovery.Endpoint) ast.Stmt {
	urlExpr := &ast.BasicLit{Kind: token.STRING, Value: fmt.Sprintf(`"/api/2.0%s"`, endpoint.GetFiberPath())}

	// input := &protos.SearchExperiments
	inputExpr := mkAssignStmt(
		[]ast.Expr{ast.NewIdent("input")},
		[]ast.Expr{
			mkAmpExpr(&ast.CompositeLit{
				Type: mkSelectorExpr(method.PackageName, method.Input),
			}),
		})

	// if err := ctx.QueryParser(&input); err != nil {
	var extractModel ast.Expr
	if endpoint.Method == "GET" {
		extractModel = mkCallExpr(mkSelectorExpr("ctx", "QueryParser"), ast.NewIdent("input"))
	} else {
		extractModel = mkCallExpr(mkSelectorExpr("ctx", "BodyParser"), ast.NewIdent("input"))
	}

	// validationErrors := service.Validate(input)
	validationErrors := mkAssignStmt(
		[]ast.Expr{ast.NewIdent("validationErrors")},
		[]ast.Expr{mkCallExpr(mkSelectorExpr("service", "Validate"), ast.NewIdent("input"))},
	)

	// if len(validationErrors) > 0 { return &fiber.Error{} )
	validationErrorsCheck := mkIfStmt(
		nil,
		mkBinaryExpr(
			mkCallExpr(ast.NewIdent("len"), ast.NewIdent("validationErrors")),
			token.GTR,
			ast.NewIdent("0"),
		),
		mkBlockStmt(
			mkReturnStmt(
				mkAmpExpr(
					&ast.CompositeLit{
						Type: mkSelectorExpr("fiber", "Error"),
						Elts: []ast.Expr{
							mkKeyValueExpr("Code", mkSelectorExpr("fiber.ErrBadRequest", "Code")),
							mkKeyValueExpr(
								"Message",
								mkCallExpr(
									mkSelectorExpr("strings", "Join"),
									ast.NewIdent("validationErrors"),
									&ast.BasicLit{Value: `" and "`})),
						},
					},
				))),
	)

	inputErrorCheck := mkIfStmt(mkAssignStmt([]ast.Expr{ast.NewIdent("err")}, []ast.Expr{extractModel}), errNotEqualNil, mkBlockStmt(returnErr))

	// output, err := service.SearchExperiments(input)
	outputExpr := mkAssignStmt([]ast.Expr{
		ast.NewIdent("output"),
		ast.NewIdent("err"),
	}, []ast.Expr{
		mkCallExpr(mkSelectorExpr("service", strcase.ToCamel(method.Name)), ast.NewIdent("input")),
	})

	// if err != nil && err.ErrorCode == protos.ErrorCode_NOT_IMPLEMENTED {
	//     return ctx.Next()
	// }
	notImplentedCheck := mkIfStmt(
		nil,
		errNotEqualNil,
		mkBlockStmt(
			mkIfStmt(
				nil,
				mkBinaryExpr(
					mkSelectorExpr("err", "ErrorCode"),
					token.EQL,
					mkSelectorExpr("protos", "ErrorCode_NOT_IMPLEMENTED"),
				),
				mkBlockStmt(mkReturnStmt(mkCallExpr(mkSelectorExpr("ctx", "Next")))),
			),
			mkReturnStmt(ast.NewIdent("err"))),
	)

	// return ctx.JSON(&output)
	returnExpr := mkReturnStmt(mkCallExpr(mkSelectorExpr("ctx", "JSON"), mkAmpExpr(ast.NewIdent("output"))))

	// func(ctx *fiber.Ctx) error { .. }
	funcExpr := &ast.FuncLit{
		Type: &ast.FuncType{
			Params: &ast.FieldList{
				List: []*ast.Field{
					mkNamedField("ctx", mkStarExpr(mkSelectorExpr("fiber", "Ctx"))),
				},
			},
			Results: &ast.FieldList{
				List: []*ast.Field{
					mkField(ast.NewIdent("error")),
				},
			},
		},
		Body: &ast.BlockStmt{
			List: []ast.Stmt{
				inputExpr,
				inputErrorCheck,
				validationErrors,
				validationErrorsCheck,
				outputExpr,
				notImplentedCheck,
				returnExpr,
			},
		},
	}

	return &ast.ExprStmt{
		// app.Get("/api/v2.0/mlflow/experiments/search", func(ctx *fiber.Ctx) error { .. })
		X: mkCallExpr(
			mkSelectorExpr("app", strcase.ToCamel(endpoint.Method)), urlExpr, funcExpr,
		),
	}
}

func mkRouteRegistrationFunction(serviceInfo discovery.ServiceInfo) *ast.FuncDecl {
	routes := make([]ast.Stmt, 0, len(serviceInfo.Methods))

	for _, method := range serviceInfo.Methods {
		for _, endpoint := range method.Endpoints {
			endpointName := fmt.Sprintf("%s_%s", serviceInfo.Name, method.Name)
			if _, ok := ImplementedEndpoints[endpointName]; ok {
				routes = append(routes, mkAppRoute(method, endpoint))
			}

		}
	}

	return &ast.FuncDecl{
		Name: ast.NewIdent(fmt.Sprintf("Register%sRoutes", serviceInfo.Name)),
		Type: &ast.FuncType{
			Params: &ast.FieldList{
				List: []*ast.Field{
					mkNamedField("service", ast.NewIdent(serviceInfo.Name)),
					mkNamedField("app", mkStarExpr(ast.NewIdent("fiber.App"))),
				},
			},
		},
		Body: &ast.BlockStmt{
			List: routes,
		},
	}
}

// Generate the service interface and route registration functions
func generateServices(pkgFolder string) error {
	decls := []ast.Decl{importStatements}

	services := discovery.GetServiceInfos()
	for _, serviceInfo := range services {
		decls = append(decls, mkServiceInterfaceNode(serviceInfo))
	}

	for _, serviceInfo := range services {
		decls = append(decls, mkRouteRegistrationFunction(serviceInfo))
	}

	// Set up the FileSet and the AST File
	fset := token.NewFileSet()

	pkg := "contract"

	file := &ast.File{
		Name:  ast.NewIdent(pkg),
		Decls: decls,
	}

	outputPath := filepath.Join(pkgFolder, pkg, "interface.g.go")

	return saveASTToFile(fset, file, true, outputPath)
}

var jsonFieldTagRegexp = regexp.MustCompile(`json:"([^"]+)"`)

// Inspect the AST of the incoming file and add a query annotation to the struct tags which have a json tag.
func addQueryAnnotation(generatedGoFile string) error {
	// Parse the file into an AST
	fset := token.NewFileSet()
	node, err := parser.ParseFile(fset, generatedGoFile, nil, parser.ParseComments)
	if err != nil {
		panic(err)
	}

	// Create an AST inspector to modify specific struct tags
	ast.Inspect(node, func(n ast.Node) bool {
		// Look for struct type declarations
		ts, ok := n.(*ast.TypeSpec)
		if !ok {
			return true
		}
		st, ok := ts.Type.(*ast.StructType)
		if !ok {
			return true
		}

		// Iterate over fields in the struct
		for _, field := range st.Fields.List {
			if field.Tag == nil {
				continue
			}
			tagValue := field.Tag.Value

			hasQuery := strings.Contains(tagValue, "query:")
			hasValidate := strings.Contains(tagValue, "validate:")
			validationKey := fmt.Sprintf("%s_%s", ts.Name, field.Names[0])
			validationRule, needsValidation := Validations[validationKey]

			if hasQuery && (!needsValidation || needsValidation && hasValidate) {
				continue
			}

			// With opening ` tick
			newTag := tagValue[0 : len(tagValue)-1]

			matches := jsonFieldTagRegexp.FindStringSubmatch(tagValue)
			if len(matches) > 0 && !hasQuery {
				// Modify the tag here
				// The json annotation could be something like `json:"key,omitempty"`
				// We only want the key part, the `omitempty` is not relevant for the query annotation
				key := matches[1]
				if strings.Contains(key, ",") {
					key = strings.Split(key, ",")[0]
				}
				// Add query annotation
				newTag += fmt.Sprintf(" query:\"%s\"", key)
			}

			if needsValidation {
				// Add validation annotation
				newTag += fmt.Sprintf(" validate:\"%s\"", validationRule)
			}

			// Closing ` tick
			newTag += "`"
			field.Tag.Value = newTag
		}
		return false
	})

	return saveASTToFile(fset, node, false, generatedGoFile)
}

func addQueryAnnotations(pkgFolder string) error {
	protoFolder := filepath.Join(pkgFolder, "protos")

	if _, pathError := os.Stat(protoFolder); os.IsNotExist(pathError) {
		return fmt.Errorf("the %s folder does not exist. Are the Go protobuf files generated?", protoFolder)
	}

	err := filepath.WalkDir(protoFolder, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if filepath.Ext(path) == ".go" {
			err = addQueryAnnotation(path)
		}

		return err
	})

	return err
}

func main() {
	if len(os.Args) != 2 {
		fmt.Println("Usage: program <path to mlflow/go/pkg folder>")
		os.Exit(1)
	}

	pkgFolder := os.Args[1]
	if _, err := os.Stat(pkgFolder); os.IsNotExist(err) {
		fmt.Printf("The provided path does not exist: %s\n", pkgFolder)
		os.Exit(1)
	}

	if err := addQueryAnnotations(pkgFolder); err != nil {
		fmt.Printf("Error adding query annotations: %s\n", err)
		os.Exit(1)
	}

	if err := generateServices(pkgFolder); err != nil {
		fmt.Printf("Error generating services: %s\n", err)
		os.Exit(1)
	}

	fmt.Println("Successfully added query annotations and generated services!")
}
